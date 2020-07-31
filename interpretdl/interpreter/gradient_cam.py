import typing
from typing import Any, Callable, List, Tuple, Union

import IPython.display as display
import cv2
import numpy as np
import os, sys
from PIL import Image

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image
from ..data_processor.visualizer import visualize_gradcam


class GradCAMInterpreter(Interpreter):
    """
    Gradient CAM Interpreter.

    More details regarding the GradCAM method can be found in the original paper:
    https://arxiv.org/abs/1610.02391
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

            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            gradcam = GradCAMInterpreter(paddle_model, "assets/ResNet50_pretrained",True)
            gradcam.interpret(
                    'assets/catdog.png',
                    'res5c.add.output.5.tmp_0',
                    label=None,
                    visual=True,
                    save_path='gradcam_test.jpg')
        """

        # Read in image
        if isinstance(data, str):
            with open(data, 'rb') as f:
                org = Image.open(f)
                org = org.convert('RGB')
                org = np.array(org)
            img = read_image(data, crop_size=self.model_input_shape[1])
            data = preprocess_image(img)
        else:
            org = data.copy

        self.target_layer_name = target_layer_name
        self.label = label

        if not self.paddle_prepared:
            self._paddle_prepare()

        feature_map, gradients = self.predict_fn(data)

        f = np.array(feature_map)[0]
        g = np.array(gradients)[0]

        visualize_gradcam(f, g, org, visual, save_path)

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():

                    image_op = fluid.data(
                        name='image',
                        shape=[1] + self.model_input_shape,
                        dtype='float32')
                    label_op = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')

                    probs = self.paddle_model(image_op)
                    if isinstance(probs, tuple):
                        probs = probs[0]

                    # manually switch the model to test mode
                    for op in main_program.global_block().ops:
                        if op.type == 'batch_norm':
                            op._set_attr('use_global_stats', True)
                        elif op.type == 'dropout':
                            op._set_attr('dropout_prob', 0.0)

                    # fetch the target layer
                    trainable_vars = list(main_program.list_vars())
                    for v in trainable_vars:
                        if v.name == self.target_layer_name:
                            conv = v

                    class_num = probs.shape[-1]
                    one_hot = fluid.layers.one_hot(label_op, class_num)
                    one_hot = fluid.layers.elementwise_mul(probs, one_hot)
                    target_category_loss = fluid.layers.reduce_sum(one_hot)
                    # target_category_loss = - fluid.layers.cross_entropy(probs, label_op)[0]

                    # add back-propagration
                    p_g_list = fluid.backward.append_backward(
                        target_category_loss)
                    # calculate the gradients w.r.t. the target layer
                    gradients_map = fluid.gradients(target_category_loss,
                                                    conv)[0]

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data):
                # if label is None, let it be the most likely label
                if self.label is None:
                    out = exe.run(
                        main_program,
                        feed={'image': data,
                              'label': np.array([[0]])},
                        fetch_list=[probs])

                    self.label = np.argmax(out[0][0])

                feature_map, gradients = exe.run(
                    main_program,
                    feed={'image': data,
                          'label': np.array([[self.label]])},
                    fetch_list=[conv, gradients_map])
                return feature_map, gradients

        self.predict_fn = predict_fn
        self.paddle_prepared = True
