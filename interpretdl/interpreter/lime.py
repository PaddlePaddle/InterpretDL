import os
import typing
from typing import Any, Callable, List, Tuple, Union

from interpretdl.interpreter.abc_interpreter import Interpreter
from ._lime_base import LimeBase
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import show_important_parts, visualize_image

import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid


class LIMEInterpreter(Interpreter):
    """
    LIME Interpreter.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self,
                 paddle_model: Callable,
                 trained_model_path: str,
                 model_input_shape=[3, 224, 224],
                 use_cuda=True) -> None:
        """

        Args:
            paddle_model: A user-defined function that gives access to model predictions. It takes the following arguments:
                    - image_input: An image input.
                    example:
                        def paddle_model(image_input):
                            import paddle.fluid as fluid
                            class_num = 1000
                            model = ResNet50()
                            logits = model.net(input=image_input, class_dim=class_num)
                            probs = fluid.layers.softmax(logits, axis=-1)
                            return probs
            trained_model_path:
            model_input_shape:
            use_cuda:
        """

        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.model_input_shape = model_input_shape
        self.use_cuda = use_cuda
        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase()

        self.lime_intermediate_results = {}

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
        if not self.paddle_prepared:
            self._paddle_prepare()

        data_instance = read_image(data_path)

        # only one example here
        probability = self.predict_fn(data_instance)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        lime_weights = self.lime_base.interpret_instance(
            data_instance[0],
            self.predict_fn,
            interpret_class,
            num_samples=num_samples,
            batch_size=batch_size)

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

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():
                    image_op = fluid.data(
                        name='image',
                        shape=[None] + self.model_input_shape,
                        dtype='float32')
                    probs = self.paddle_model(image_input=image_op)
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

            def predict_fn(visual_images):
                images = preprocess_image(
                    visual_images
                )  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
                [result] = exe.run(main_program,
                                   fetch_list=[probs],
                                   feed={'image': images})

                return result

        self.predict_fn = predict_fn
        self.paddle_prepared = True
