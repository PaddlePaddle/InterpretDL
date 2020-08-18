import typing
from typing import Any, Callable, List, Tuple, Union

import IPython.display as display
import cv2
import numpy as np
import os, sys
from PIL import Image

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image
from ..data_processor.visualizer import visualize_heatmap


class ScoreCAMInterpreter(Interpreter):
    """
    Score CAM Interpreter.

    More details regarding the Score CAM method can be found in the original paper:
    https://arxiv.org/abs/1910.01279
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradCAMInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data inputs.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.use_cuda = use_cuda
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

    def interpret(self,
                  data,
                  target_layer_name,
                  label=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (str or numpy.ndarray): The input image filepath or numpy array.
            target_layer_name (str): The target layer to calculate gradients.
            label (int, optional): The target label to analyze. If None, the most likely label will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None

        Returns:
            None

        Example::

            import interpretdl as it
            def paddle_model(image_input):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs

            scorecam = it.ScoreCAMInterpreter(paddle_model,
                                              "assets/ResNet50_pretrained", True)
            scorecam.interpret(
                'assets/catdog.png',
                'res5c.add.output.5.tmp_0',
                label=None,
                visual=True,
                save_path='assets/scorecam_test.jpg')
        """

        if isinstance(data, str):
            with open(data, 'rb') as f:
                org = Image.open(f)
                org = org.convert('RGB')
                org = np.array(org)
                img = read_image(data, crop_size=self.model_input_shape[1])
                data = preprocess_image(img)
        else:
            org = data.copy

        b, c, h, w = data.shape

        self.target_layer_name = target_layer_name
        self.label = label

        if not self.paddle_prepared:
            self._paddle_prepare()

        if self.label is None:
            _, probs = self.predict_fn(data)
            self.label = np.argmax(probs, axis=1)

        feature_map, _ = self.predict_fn(data)
        attributions = np.zeros((1, 1, h, w))

        for i in range(feature_map.shape[1]):
            feature_channel = np.expand_dims(feature_map[:, i, :, :], 1)[0][0]
            feature_channel = cv2.resize(feature_channel, (h, w))
            norm_feature_channel = (
                feature_channel - feature_channel.min()) / (
                    feature_channel.max() - feature_channel.min())
            _, probs = self.predict_fn(data * norm_feature_channel)
            score = probs[0][self.label]
            attributions += score * feature_channel

        attributions = np.maximum(attributions, 0)
        attributions_min, attributions_max = attributions.min(
        ), attributions.max()

        if attributions_min == attributions_max:
            return None

        interpretations = (attributions - attributions_min) / (
            attributions_max - attributions_min)

        visualize_heatmap(interpretations[0][0], org, visual, save_path)

        return interpretations

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():
                    data_op = fluid.data(
                        name='data',
                        shape=[1] + self.model_input_shape,
                        dtype='float32')
                    probs = self.paddle_model(data_op)
                    if isinstance(probs, tuple):
                        probs = probs[0]
                    trainable_vars = list(main_program.list_vars())
                    for v in trainable_vars:
                        if v.name == self.target_layer_name:
                            conv = v

                    main_program = main_program.clone(for_test=True)

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data):
                feature_map, probs_out = exe.run(main_program,
                                                 feed={'data': data},
                                                 fetch_list=[conv, probs])
                return feature_map, probs_out

        self.predict_fn = predict_fn
        self.paddle_prepared = True
