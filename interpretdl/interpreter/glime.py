import numpy as np
import paddle

from .lime import LIMECVInterpreter
from ._lime_base import compute_segments
from ._global_prior_base import get_cluster_label, cluster_global_weights_to_local_prior
from ..data_processor.readers import preprocess_image, read_image, restore_image
from ..data_processor.readers import load_npy_dict_file
from ..data_processor.visualizer import sp_weights_to_image_explanation, overlay_threshold, save_image, show_vis_explanation


class GLIMECVInterpreter(LIMECVInterpreter):
    """
    G-LIME CV Interpreter. This is an Interpreter in progress.
    """

    def __init__(self, paddle_model: callable, device: str = 'gpu:0') -> None:
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        LIMECVInterpreter.__init__(self, paddle_model, device)
        self.global_weights = None

    def set_global_weights(self, global_weights_info: str or dict):
        """Set directly the global weights without any pre-computations.

        Args:
            global_weights_info (str or dict): A path of the file or the dict.
        """
        if isinstance(global_weights_info, str):
            self.global_weights = load_npy_dict_file(global_weights_info)
        elif isinstance(global_weights_info, dict):
            self.global_weights = global_weights_info
        else:
            print("Warning: Not set global weights. Unknown type.")
            return

        print(f"Set Global Weights from {global_weights_info}")

    def compute_global_weights(self,
                               g_name: str = 'normlime',
                               list_of_lime_explanations: list = None,
                               list_file_paths: list = None,
                               save_path: str = None):
        """Compute the global weights, given the ``list_of_lime_explanations``. This is done by NormLIME or Average 
        Global Explanations, which are introduced in https://arxiv.org/abs/1909.04200 and 
        https://arxiv.org/abs/1907.03039 respectively.

        Args:
            g_name (str, optional): The method to aggregate local explanations. Defaults to ``'normlime'``.
            list_of_lime_explanations (list, optional): The LIME results. Defaults to None.
            list_file_paths (list, optional): This is not implemented currently. Defaults to None.
            save_path (str, optional): A path to save the global weights, which can be directly used the next time, 
                and called by ``set_global_weights()``. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError. 

        Returns:
            dict: Global Weights.
        """
        if list_file_paths is not None:
            raise NotImplementedError("Use scripts/benchmark.py to compute LIME explanations.")

        # check the first one
        assert 'input' in list_of_lime_explanations[0]
        assert 'lime_weights' in list_of_lime_explanations[0]
        assert 'segmentation' in list_of_lime_explanations[0]

        global_weights_all_labels = {}
        for lime_explanation in list_of_lime_explanations:
            cluster_labels = get_cluster_label(lime_explanation['input'][np.newaxis, ...],
                                               lime_explanation['segmentation'])

            pred_labels = lime_explanation['lime_weights'].keys()

            for y in pred_labels:
                global_weights_y = global_weights_all_labels.get(y, {})
                w_f_y = [abs(w[1]) for w in lime_explanation['lime_weights'][y]]
                w_f_y_l1norm = sum(w_f_y)

                for w in lime_explanation['lime_weights'][y]:
                    seg_label = w[0]
                    if g_name == 'normlime':
                        weight = w[1] * w[1] / w_f_y_l1norm
                    elif g_name == 'avg':
                        weight = abs(w[1])
                    else:
                        weight = w[1] * w[1]

                    tmp = global_weights_y.get(cluster_labels[seg_label], [])
                    tmp.append(weight)
                    global_weights_y[cluster_labels[seg_label]] = tmp

                global_weights_all_labels[y] = global_weights_y

        # compute global weights.
        for y in global_weights_all_labels:
            global_weights_y = global_weights_all_labels.get(y, {})
            for k in global_weights_y:
                global_weights_y[k] = sum(global_weights_y[k]) / len(global_weights_y[k])

        if save_path is not None:
            print(f"Saving Global Weights to {save_path}")
            np.save(save_path, global_weights_all_labels)

        self.global_weights = global_weights_all_labels
        return self.global_weights

    def interpret(self,
                  data: str,
                  interpret_class: int or None = None,
                  top_k: int = 1,
                  prior_method: str = 'none',
                  prior_reg_force: float = 1.0,
                  num_samples: int = 1000,
                  batch_size: int = 50,
                  resize_to: int = 224,
                  crop_to: int = None,
                  visual: bool = True,
                  save_path: str = None):
        """
        Note that for GLIME interpreter, :py:func:`set_global_weights()` needs to be called before calling 
        :py:func:`interpret()`. Basically, the technical process of GLIME is similar to LIME. See the
        `tutorial
        <https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/LIME_Variants_part2.ipynb>`_ for more 
        details.

        Args:
            data (str): The input file path.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be 
                used. Default: ``None``.
            top_k (int, optional): Number of top classes to interpret. Will not be used if ``interpret_class`` is 
                given. Default: ``1``.
            prior_method: Prior method. Can be chosen from ``{"none", "ridge"}``. Defaults to ``"none"``, which is 
                equivalent to LIME. If ``none``, :py:func:`interpret()` will use zeros as prior; Otherwise, the loaded
                prior will be used.
            prior_reg_force (float, optional): The regularization force to apply. Default: ``1.0``.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate 
                interpretation. Default: ``1000``.
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to
                ``224``.
            crop_to ([type], optional): After resize, images will be center cropped to a square image 
                with the size ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The path to save the processed image. If None, the image will not be saved. 
                Default: ``None``.

        Returns:
            [dict]: LIME results: {interpret_label_i: weights on features}

        """
        if self.global_weights is None and prior_method != "none":
            raise ValueError(
                "The interpreter is not prepared. Call compute_global_weights() or set_global_weights()"\
                    " before interpretation."
            )

        # preprocess_input
        if isinstance(data, str):
            img = read_image(data, resize_to, crop_to)
        else:
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0)
            if np.issubdtype(data.dtype, np.integer):
                img = data
            else:
                img = restore_image(data.copy())  # for later visualization
        data = preprocess_image(img)

        self._build_predict_fn(output='probability')  # create self.predict_fn.
        probability, _, _ = self.predict_fn(data, None)
        probability = probability[0]  # only one example here

        if interpret_class is None:
            # only interpret top 1 if not provided.
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-top_k:]
            interpret_class = np.array(interpret_class)
        elif isinstance(interpret_class, list):
            interpret_class = np.array(interpret_class)
        else:
            interpret_class = np.array([interpret_class])

        def predict_fn_for_lime(_imgs):
            _data = preprocess_image(_imgs)  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]

            output, _, _ = self.predict_fn(_data, None)
            return output

        self.predict_fn_for_lime = predict_fn_for_lime

        if self.lime_base.segments is None:
            self.lime_base.segments = compute_segments(img[0])

        segments = self.lime_base.segments

        if prior_method == "none":
            prior = np.zeros(len(np.unique(segments)))
        else:
            prior = cluster_global_weights_to_local_prior(img, segments, interpret_class[0], self.global_weights)

        lime_weights, r2_scores = self.lime_base.interpret_instance(img[0],
                                                                    self.predict_fn_for_lime,
                                                                    interpret_class,
                                                                    num_samples=num_samples,
                                                                    batch_size=batch_size,
                                                                    prior=prior,
                                                                    reg_force=prior_reg_force)

        # visualization and save image.
        explanation_mask = sp_weights_to_image_explanation(img[0], lime_weights, interpret_class[0],
                                                           self.lime_base.segments)
        explanation_vis = overlay_threshold(img[0], explanation_mask)
        if visual:
            show_vis_explanation(explanation_vis)
        if save_path is not None:
            save_image(save_path, explanation_vis)

        self.lime_results['probability'] = {c: probability[c] for c in interpret_class.ravel()}
        self.lime_results['input'] = img[0]
        self.lime_results['segmentation'] = self.lime_base.segments
        self.lime_results['r2_scores'] = r2_scores
        self.lime_results['lime_weights'] = lime_weights

        return lime_weights
