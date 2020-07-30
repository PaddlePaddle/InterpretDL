import os
import typing
from typing import Any, Callable, List, Tuple, Union

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image
from ..data_processor.visualizer import visualize_grayscale

import matplotlib.pyplot as plt
import numpy as np


class OcclusionInterpreter(Interpreter):
    """
    Occlusion Interpreter.

    More details regarding the Occlusion method can be found in the original paper:
    https://arxiv.org/abs/1311.2901
    """

    def __init__(self,
                 paddle_model: Callable,
                 trained_model_path: str,
                 model_input_shape=[3, 224, 224],
                 use_cuda=True) -> None:
        """
        Initialize the OcclusionInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions. It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.

            trained_model_path (str): The pretrained model directory.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """

        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.model_input_shape = model_input_shape
        self.use_cuda = use_cuda
        self.paddle_prepared = False

    def interpret(self,
                  data,
                  sliding_window_shapes,
                  interpret_class=None,
                  strides=1,
                  baseline=None,
                  perturbations_per_eval=1,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (str or numpy.ndarray): The image filepath or processed image.
            sliding_window_shapes (tuple): Shape of sliding windows to occlude data.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be used.
            strides (int or tuple): The step by which the occlusion should be shifted by in each direction for each iteration.
                                    If int, the step size in each direction will be the same. Default: 1
            baseline (numpy.ndarray, optional): The reference values that replace occlusioned features. Default: None
            perturbations_per_eval (int, optional): number of occlusions in each batch. Default: 1
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The path to save the processed image. If None, the image will not be saved. Default: None

        Returns:
            numpy.ndarray: total_attrib

        Example::

            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            oc = OcclusionInterpreter(paddle_model, "assets/ResNet50_pretrained")
            attributions = oc.interpret(
                    'assets/catdog.png',
                    sliding_window_shapes=(1, 30, 30),
                    interpret_class=None,
                    strides=(1, 10, 10),
                    baseline=None,
                    perturbations_per_eval=5,
                    visual=True,
                    save_path='occlusion_gray.jpg')
        """
        if not self.paddle_prepared:
            self._paddle_prepare()

        if isinstance(data, str):
            data = read_image(data)
            data = preprocess_image(data)

        if baseline is None:
            baseline = np.zeros_like(data)

        probability = self.predict_fn(data)[0]
        sliding_window = np.ones(sliding_window_shapes)

        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        current_shape = np.subtract(self.model_input_shape,
                                    sliding_window_shapes)
        shift_counts = tuple(
            np.add(np.ceil(np.divide(current_shape, strides)).astype(int), 1))
        initial_eval = probability[interpret_class]
        total_attrib = np.zeros(self.model_input_shape)

        for (ablated_features, current_mask) in self._ablation_generator(
                data, sliding_window, strides, baseline, shift_counts,
                perturbations_per_eval):

            modified_eval = self.predict_fn(np.float32(
                ablated_features))[:, interpret_class]
            eval_diff = initial_eval - modified_eval
            for i in range(eval_diff.shape[0]):
                total_attrib += (eval_diff[i] * current_mask[i])

        visualize_grayscale(
            np.array([total_attrib]), visual=visual, save_path=save_path)

        return total_attrib

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():
                    image_op = fluid.data(
                        name='image',
                        shape=[None] + self.model_input_shape,
                        dtype='float32')
                    probs = self.paddle_model(image_op)
                    if isinstance(probs, tuple):
                        probs = probs[0]
                    main_program = main_program.clone(for_test=True)

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(images):

                [result] = exe.run(main_program,
                                   fetch_list=[probs],
                                   feed={'image': images})

                return result

        self.predict_fn = predict_fn
        self.paddle_prepared = True

    def _ablation_generator(self, inputs, sliding_window, strides, baseline,
                            shift_counts, perturbations_per_eval):
        num_features = np.prod(shift_counts)
        perturbations_per_eval = min(perturbations_per_eval, num_features)
        num_features_processed = 0

        if perturbations_per_eval > 1:
            all_features_repeated = np.repeat(
                inputs, perturbations_per_eval, axis=0)
        else:
            all_features_repeated = inputs

        while num_features_processed < num_features:
            current_num_ablated_features = min(
                perturbations_per_eval, num_features - num_features_processed)
            if current_num_ablated_features != perturbations_per_eval:
                current_features = all_features_repeated[:
                                                         current_num_ablated_features]
            else:
                current_features = all_features_repeated

            ablated_features, current_mask = self._construct_ablated_input(
                current_features, baseline, num_features_processed,
                num_features_processed + current_num_ablated_features,
                sliding_window, strides, shift_counts)

            yield ablated_features, current_mask
            num_features_processed += current_num_ablated_features

    def _construct_ablated_input(self, inputs, baseline, start_feature,
                                 end_feature, sliding_window, strides,
                                 shift_counts):
        input_mask = np.array([
            self._occlusion_mask(inputs, j, sliding_window, strides,
                                 shift_counts)
            for j in range(start_feature, end_feature)
        ])
        ablated_tensor = inputs * (np.ones(input_mask.shape) - input_mask
                                   ) + baseline * input_mask

        return ablated_tensor, input_mask

    def _occlusion_mask(self, inputs, ablated_feature_num, sliding_window,
                        strides, shift_counts):
        remaining_total = ablated_feature_num

        current_index = []
        for i, shift_count in enumerate(shift_counts):
            stride = strides[i] if isinstance(strides, tuple) else strides
            current_index.append((remaining_total % shift_count) * stride)
            remaining_total = remaining_total // shift_count

        remaining_padding = np.subtract(
            inputs.shape[1:], np.add(current_index, sliding_window.shape))

        slicers = []
        for i, p in enumerate(remaining_padding):
            # When there is no enough space for sliding window, truncate the window
            if p < 0:
                slicer = [slice(None)] * len(sliding_window.shape)
                slicer[i] = range(sliding_window.shape[i] + p)
                slicers.append(slicer)

        pad_values = tuple(
            tuple(reversed(np.maximum(pair, 0)))
            for pair in zip(remaining_padding, current_index))

        for slicer in slicers:
            sliding_window = sliding_window[tuple(slicer)]
        padded = np.pad(sliding_window, pad_values)

        return padded
