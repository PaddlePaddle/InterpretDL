import numpy as np

from ..data_processor.readers import preprocess_image, read_image, restore_image
from ..data_processor.visualizer import sp_weights_to_image_explanation, overlay_threshold, save_image, show_vis_explanation

from ._lime_base import LimeBase
from .abc_interpreter import Interpreter, InputOutputInterpreter


class LIMECVInterpreter(InputOutputInterpreter):
    """
    LIME presents a locally explanation by fitting a set of perturbed samples near the target sample using an 
    interpretable model, specifically a linear model. 

    The implementation is based on https://github.com/marcotcr/lime.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938.
    """

    def __init__(self,
                 paddle_model: callable,
                 device: str = 'gpu:0',
                 use_cuda: bool = None,
                 random_seed: int or None = None):
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        InputOutputInterpreter.__init__(self, paddle_model, device, use_cuda)

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed)
        self.lime_results = {}

    def interpret(self,
                  data: str,
                  interpret_class: int = None,
                  num_samples: int = 1000,
                  batch_size: int = 50,
                  resize_to: int = 224,
                  crop_to: int = None,
                  visual: bool = True,
                  save_path: str = None):
        """
        Main function of the interpreter.

        The implementation is based on https://github.com/marcotcr/lime.

        Args:
            data (str): The input file path.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be 
                used. Default: ``None``.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate 
                interpretation. Default: ``1000``.
            batch_size (int, optional): Number of samples to forward each time. Default: ``50``.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The filepath(s) to save the processed image(s). If None, the image will not be 
                saved. Default: ``None``.

        Returns:
            [dict]: LIME results: {interpret_label_i: weights on features}
        """
        # preprocess_inputs
        if isinstance(data, str):
            crop_size = crop_to
            target_size = resize_to
            img = read_image(data, target_size, crop_size)
        else:
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0)
            if np.issubdtype(data.dtype, np.integer):
                img = data
            else:
                # for later visualization
                img = restore_image(data.copy())
        data = preprocess_image(img)
        data_type = np.array(data).dtype
        self.data_type = data_type

        self._build_predict_fn(output='probability')

        probability, _ = self.predict_fn(data, None)
        # only one example here
        probability = probability[0]

        if interpret_class is None:
            # only interpret top 1 if not provided.
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]
            interpret_class = np.array(interpret_class)
        elif isinstance(interpret_class, list):
            interpret_class = np.array(interpret_class)
        else:
            interpret_class = np.array([interpret_class])

        def predict_fn_for_lime(_imgs):
            _data = preprocess_image(_imgs)  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]

            output, _ = self.predict_fn(_data, None)
            return output

        self.predict_fn_for_lime = predict_fn_for_lime
        lime_weights, r2_scores = self.lime_base.interpret_instance(img[0],
                                                                    self.predict_fn_for_lime,
                                                                    interpret_class,
                                                                    num_samples=num_samples,
                                                                    batch_size=batch_size)

        # visualization and save image.
        if save_path is None and not visual:
            # no need to visualize or save explanation results.
            pass
        else:
            explanation_mask = sp_weights_to_image_explanation(img[0], lime_weights, interpret_class[0],
                                                               self.lime_base.segments)
            explanation_vis = overlay_threshold(img[0], explanation_mask)
            if visual:
                show_vis_explanation(explanation_vis)
            if save_path is not None:
                save_image(save_path, explanation_vis)

        # intermediate results, for possible further usages.
        self.lime_results['probability'] = {c: probability[c] for c in interpret_class.ravel()}
        self.lime_results['input'] = img[0]
        self.lime_results['segmentation'] = self.lime_base.segments
        self.lime_results['r2_scores'] = r2_scores
        self.lime_results['lime_weights'] = lime_weights

        return lime_weights


class LIMENLPInterpreter(Interpreter):
    """
    LIME Interpreter for NLP tasks.

    LIME presents a locally explanation by fitting a set of perturbed samples near the target sample using an 
    interpretable model, specifically a linear model. 

    The implementation is based on https://github.com/marcotcr/lime.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938.

    """

    def __init__(self,
                 paddle_model: callable,
                 device: str = 'gpu:0',
                 use_cuda=None,
                 random_seed: int or None = None) -> None:
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
            random_seed (int): random seed. Defaults to None.
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda)
        self.paddle_model = paddle_model
        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed)

        self.lime_intermediate_results = {}

    def interpret(self,
                  data: str,
                  preprocess_fn: callable,
                  unk_id: int,
                  pad_id: int or None = None,
                  interpret_class: int = None,
                  num_samples: int = 1000,
                  batch_size: int = 50,
                  lod_levels: int = None,
                  return_pred: bool = False,
                  visual: bool = True):
        """
        Main function of the interpreter.

        The implementation is based on https://github.com/marcotcr/lime.

        Args:
            data (str): The raw string for analysis.
            preprocess_fn (callable): A user-defined function that input raw string and outputs the a tuple of inputs 
                to feed into the NLP model.
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. 
                Default: ``None``.
            interpret_class (list or numpy.ndarray, optional): The index of class to interpret. If None, the most 
                likely label will be used. Default: ``None``.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate
                interpretation. Default: ``1000``.
            batch_size (int, optional): Number of samples to forward each time. Default: ``50``.
            lod_levels (list or tuple or numpy.ndarray or None, optional): The lod levels for model inputs. It should
                have the length equal to number of outputs given by preprocess_fn. If None, lod levels are all zeros. 
                Default: ``None``.
            visual (bool, optional): Whether or not to visualize. Default: ``True``.

        Returns:
            [dict]: LIME results: {interpret_label_i: weights on features}
        """

        model_inputs = preprocess_fn(data)
        if not isinstance(model_inputs, tuple):
            self.model_inputs = (np.array(model_inputs), )
        else:
            self.model_inputs = tuple(inp.numpy() for inp in model_inputs)

        if not self.paddle_prepared:
            self._paddle_prepare()
        # only one example here
        probability = self.predict_fn(*self.model_inputs)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        lime_weights, r2_scores = self.lime_base.interpret_instance_text(self.model_inputs,
                                                                         classifier_fn=self.predict_fn,
                                                                         interpret_labels=interpret_class,
                                                                         unk_id=unk_id,
                                                                         pad_id=pad_id,
                                                                         num_samples=num_samples,
                                                                         batch_size=batch_size)

        data_array = self.model_inputs[0]
        data_array = data_array.reshape((np.prod(data_array.shape), ))
        for c in lime_weights:
            weights_c = lime_weights[c]
            weights_new = [(data_array[tup[0]], tup[1]) for tup in weights_c]
            lime_weights[c] = weights_new

        # Visualization is currently not supported here.
        # See the tutorial for more information:
        # https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb
        if return_pred:
            return (interpret_class, probability[interpret_class], lime_weights)
        return lime_weights

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle
            if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
                print("Paddle is not installed with GPU support. Change to CPU version now.")
                self.device = 'cpu'
            paddle.set_device(self.device)
            self.paddle_model.eval()

            def predict_fn(*params):
                params = tuple(paddle.to_tensor(inp) for inp in params)
                logits = self.paddle_model(*params)
                probs = paddle.nn.functional.softmax(logits)
                return probs.numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True
