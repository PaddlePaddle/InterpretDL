import os
import typing
from typing import Any, Callable, List, Tuple, Union

from .lime import LIMEInterpreter
from ._lime_base import compute_segments
from interpretdl.data_processor.readers import preprocess_image, read_image, load_npy_dict_file
from interpretdl.data_processor.visualizer import show_important_parts, visualize_image
from ._global_prior_base import precompute_global_prior, use_fast_normlime_as_prior

import matplotlib.pyplot as plt
import numpy as np


class LIMEPriorInterpreter(LIMEInterpreter):
    def __init__(self,
                 paddle_model: Callable,
                 trained_model_path: str,
                 model_input_shape=[3, 224, 224],
                 prior_method="none",
                 use_cuda=True) -> None:
        """

        Args:
            paddle_model:
            trained_model_path:
            model_input_shape:
            prior_method:
            use_cuda:
        """
        LIMEInterpreter.__init__(self, paddle_model, trained_model_path,
                                 model_input_shape, use_cuda)
        self.prior_method = prior_method
        self.global_weights = None

    def interpreter_init(self,
                         list_file_paths=None,
                         batch_size=0,
                         weights_file_path=None,
                         prior_reg_force=1.0):
        """
        Pre-compute global weights.
        If `weights_file_path` is given and has contents, then skip the pre-compute process and
        read the content as the global weights; else the pre-compute process should be performed
        with `list_file_paths`. After the pre-compute process, if `weights_file_path` is given,
        then the prec-computed weights will be saved in this path for skipping the pre-compute
        process in the next usage.
        This is an additional step that LIMEPrior Interpreter has to perform before interpretation.

        Args:
            list_file_paths:
            batch_size:
            weights_file_path:

        Returns:

        """

        assert list_file_paths is not None or weights_file_path is not None, \
            "Cannot prepare without anything. "

        if not self.paddle_prepared:
            self._paddle_prepare()

        self.prior_reg_force = prior_reg_force
        precomputed_weights = load_npy_dict_file(weights_file_path)
        if precomputed_weights is not None:
            self.global_weights = precomputed_weights
        else:
            self.global_weights = precompute_global_prior(
                list_file_paths, self.predict_fn, batch_size,
                self.prior_method)
            if weights_file_path is not None:
                np.save(weights_file_path, self.global_weights)

    def interpret(self,
                  data_path,
                  interpret_class=None,
                  num_samples=1000,
                  batch_size=50,
                  visual=True,
                  save_path=None):
        """

        Args:
            data_path: The input file path.
            interpret_class: The index of class to interpret. If None, the most likely label will be used.
            num_samples: LIME sampling numbers. Larger number of samples usually gives more accurate interpretation.
            batch_size: Number of samples to forward each time.
            visual: Whether or not to visualize the processed image.
            save_path: The path to save the processed image. If None, the image will not be saved.

        Returns:
            lime_weights: a dict {interpret_label_i: weights on features}

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

        lime_weights = self.lime_base.interpret_instance(
            data_instance[0],
            self.predict_fn,
            interpret_class,
            num_samples=num_samples,
            batch_size=batch_size,
            prior=prior,
            reg_force=self.prior_reg_force)

        interpretation = show_important_parts(data_instance[0], lime_weights,
                                              interpret_class[0],
                                              self.lime_base.segments)

        if visual:
            visualize_image(interpretation)

        if save_path is not None:
            plt.imsave(save_path, interpretation)

        self.lime_intermediate_results['probability'] = probability
        self.lime_intermediate_results['input'] = data_instance[0]
        self.lime_intermediate_results[
            'segmentation'] = self.lime_base.segments

        return lime_weights
