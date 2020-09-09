import os
import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np

from .lime import LIMECVInterpreter
from ._lime_base import compute_segments
from ._global_prior_base import precompute_global_prior, use_fast_normlime_as_prior
from ..data_processor.readers import read_image, load_npy_dict_file
from ..data_processor.visualizer import show_important_parts, visualize_image, save_image


class LIMEPriorInterpreter(LIMECVInterpreter):
    """
    LIME Prior Interpreter.
    """

    def __init__(self,
                 paddle_model: Callable,
                 trained_model_path: str,
                 model_input_shape=[3, 224, 224],
                 prior_method="none",
                 use_cuda=True) -> None:
        """
        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data input.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory. It will be loaded by ``fluid.io.load_persistables``.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            prior_method: Prior method. Can be chosen from ``{"none", "ridge"}``.
                Defaults to ``"none"``, which is equivalent to LIME.
                If ``"none"``, ``interpret()`` will use zeros as prior;
                Otherwise, the loaded prior will be used.
            use_cuda: Whether to use CUDA. Defaults to ``True``.
        """
        LIMECVInterpreter.__init__(self, paddle_model, trained_model_path,
                                   model_input_shape, use_cuda)
        self.prior_method = prior_method
        self.global_weights = None

    def interpreter_init(self,
                         list_file_paths=None,
                         batch_size=0,
                         weights_file_path=None):
        """
        Pre-compute global weights.

        If ``weights_file_path`` is given and has contents, then skip the pre-compute process and
        read the content as the global weights; else the pre-compute process should be performed
        with `list_file_paths`. After the pre-compute process, if `weights_file_path` is given,
        then the prec-computed weights will be saved in this path for skipping the pre-compute
        process in the next usage.
        This is an additional step that LIMEPrior Interpreter has to perform before interpretation.

        Args:
            list_file_paths: List of files used to compute the global prior.
            batch_size: Number of samples to forward each time.
            weights_file_path: Path to load as prior.

        Returns:
            None.

        """

        assert list_file_paths is not None or weights_file_path is not None, \
            "Cannot prepare without anything. "

        if not self.paddle_prepared:
            self._paddle_prepare()

        precomputed_weights = load_npy_dict_file(weights_file_path)
        if precomputed_weights is not None:
            self.global_weights = precomputed_weights
        else:
            self.global_weights = precompute_global_prior(
                list_file_paths, self.predict_fn, batch_size,
                self.prior_method)
            if weights_file_path is not None and self.global_weights is not None:
                np.save(weights_file_path, self.global_weights)

    def interpret(self,
                  data_path,
                  interpret_class=None,
                  prior_reg_force=1.0,
                  num_samples=1000,
                  batch_size=50,
                  visual=True,
                  save_path=None):
        """
        Note that for LIME prior interpreter, ``interpreter_init()`` needs to be called before calling ``interpret()``.

        Args:
            data_path (str): The input file path.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be used. Default: None
            prior_reg_force (float, optional): The regularization force to apply. Default: 1.0
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The path to save the processed image. If None, the image will not be saved. Default: None

        :return: LIME Prior weights: {interpret_label_i: weights on features}
        :rtype: dict

        Example::

            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs

            limegp = LIMEPriorInterpreter(
                paddle_model, trained_model, prior_method="ridge")
            limegp.interpreter_init(
                image_paths, batch_size=100, weights_file_path="assets/gp_weights.npy")
            lime_weights = limegp.interpret(
                image_paths[0],
                num_samples=1000,
                batch_size=100,
                save_path='assets/lime_gp.png',
                prior_reg_force=1.0)

        """
        if self.global_weights is None and self.prior_method != "none":
            raise ValueError(
                "The interpreter is not prepared. Call prepare() before interpretation."
            )

        data_instance = read_image(data_path)

        # only one example here
        probability = self.predict_fn(data_instance)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        segments = compute_segments(data_instance[0])
        if self.prior_method == "none":
            prior = np.zeros(len(np.unique(segments)))
        else:
            prior = use_fast_normlime_as_prior(data_instance, segments,
                                               interpret_class[0],
                                               self.global_weights)

        lime_weights, r2_scores = self.lime_base.interpret_instance(
            data_instance[0],
            self.predict_fn,
            interpret_class,
            num_samples=num_samples,
            batch_size=batch_size,
            prior=prior,
            reg_force=prior_reg_force)

        interpretation = show_important_parts(
            data_instance[0],
            lime_weights,
            interpret_class[0],
            self.lime_base.segments,
            visual=visual,
            save_path=save_path)

        self.lime_intermediate_results['probability'] = probability
        self.lime_intermediate_results['input'] = data_instance[0]
        self.lime_intermediate_results[
            'segmentation'] = self.lime_base.segments
        self.lime_intermediate_results['r2_scores'] = r2_scores

        return lime_weights
