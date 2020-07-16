import typing
from typing import Any, Callable, List, Tuple, Union

from interpretdl.interpreter.abc_interpreter import Interpreter
from interpretdl.data_processor.readers import preprocess_image, read_image

import IPython.display as display
import cv2
import numpy as np
import paddle.fluid as fluid
import os, sys
from PIL import Image


class GradCAMInterpreter(Interpreter):
    """
    Gradient CAM Interpreter.

    More details regarding the GradCAM method can be found in the original paper:
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 class_num,
                 target_layer_name,
                 use_cuda,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradCAMInterpreter

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
            trained_model_path: The pretrained model directory.
            class_num: Number of classes for the model.
            target_layer_name: The target layer to calculate gradients.
            use_cuda: Whether or not to use cuda.
            model_input_shape: The input shape of the model
        """
        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.class_num = class_num
        self.target_layer_name = target_layer_name
        self.use_cuda = use_cuda
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

    def interpret(self, data, label=None, visual=True, save_path=None):
        """
        Main function of the interpreter.

        Args:
            img_path: The input image filepath or numpy array.
            label: The target label to analyze. If None, the most likely label will be used.
            visual: Whether or not to visualize the processed image.
            save_path: The filepath to save the processed image. If None, the image will not be saved.

        Returns:
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

        self.label = label

        if not self.paddle_prepared:
            self._paddle_prepare()

        feature_map, gradients = self.predict_fn(data)

        f = np.array(feature_map)[0]
        g = np.array(gradients)[0]
        # take the average of gradient for each channel
        mean_g = np.mean(g, (1, 2))
        heatmap = f.transpose([1, 2, 0])
        # multiply the feature map by average gradients
        for i in range(len(mean_g)):
            heatmap[:, :, i] *= mean_g[i]

        heatmap = np.mean(heatmap, axis=-1)
        # ReLU
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        org = np.array(org).astype('float32')
        org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap, (org.shape[1], org.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        x = heatmap * 0.8 + org
        if visual:
            display.display(display.Image(x))

        if save_path is not None:
            cv2.imwrite(save_path, x)

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
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

                    # fetch the target layer
                    trainable_vars = list(main_program.list_vars())
                    for v in trainable_vars:
                        if v.name == self.target_layer_name:
                            conv = v

                    one_hot = fluid.layers.one_hot(label_op, self.class_num)
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
