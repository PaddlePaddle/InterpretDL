import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import os, sys
from PIL import Image
import paddle

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image, preprocess_inputs
from ..data_processor.visualizer import visualize_heatmap


class GradCAMInterpreter(Interpreter):
    """
    Gradient CAM Interpreter.

    More details regarding the GradCAM method can be found in the original paper:
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self,
                 paddle_model,
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
        self.use_cuda = use_cuda
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

    def interpret(self,
                  inputs,
                  target_layer_name,
                  labels=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            target_layer_name (str): The target layer to calculate gradients.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/heatmap for each image
        :rtype: numpy.ndarray

        Example::

            import interpretdl as it
            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            gradcam = it.GradCAMInterpreter(paddle_model, "assets/ResNet50_pretrained",True)
            gradcam.interpret(
                    'assets/catdog.png',
                    'res5c.add.output.5.tmp_0',
                    label=None,
                    visual=True,
                    save_path='assets/gradcam_test.jpg')
        """

        imgs, data, save_path = preprocess_inputs(inputs, save_path,
                                                  self.model_input_shape)

        self.target_layer_name = target_layer_name

        if not self.paddle_prepared:
            self._paddle_prepare()

        bsz = len(data)
        if labels is None:
            _, _, preds = self.predict_fn(data, labels)
            labels = preds
        labels = np.array(labels).reshape((bsz, ))

        feature_map, gradients, _ = self.predict_fn(data, labels)

        f = np.array(feature_map)
        g = np.array(gradients)
        mean_g = np.mean(g, (2, 3))
        heatmap = f.transpose([0, 2, 3, 1])
        dim_array = np.ones((1, heatmap.ndim), int).ravel()
        dim_array[heatmap.ndim - 1] = -1
        dim_array[0] = bsz
        heatmap = heatmap * mean_g.reshape(dim_array)

        heatmap = np.mean(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap, axis=tuple(np.arange(1, heatmap.ndim)))
        heatmap /= heatmap_max.reshape((bsz, ) + (1, ) * (heatmap.ndim - 1))
        for i in range(bsz):
            visualize_heatmap(heatmap[i], imgs[i], visual, save_path[i])

        return heatmap

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            if self.use_cuda:
                paddle.set_device('gpu:0')
            else:
                paddle.set_device('cpu')

            self.paddle_model.train()

            feature_maps = None

            def hook(layer, input, output):
                global feature_maps
                feature_maps = output

            for n, v in self.paddle_model.named_sublayers():
                if n == self.target_layer_name:
                    v.register_forward_post_hook(hook)
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0

            def predict_fn(data, labels):
                global feature_maps
                data = paddle.to_tensor(data)
                data.stop_gradient = False
                out = self.paddle_model(data)
                out = paddle.nn.functional.softmax(out, axis=1)
                preds = paddle.argmax(out, axis=1)
                if labels is None:
                    labels = preds.numpy()
                labels_onehot = paddle.nn.functional.one_hot(
                    paddle.to_tensor(labels), num_classes=out.shape[1])
                target = paddle.sum(out * labels_onehot, axis=1)
                gradients = paddle.grad(
                    outputs=[target], inputs=[feature_maps])[0]
                return feature_maps.numpy(), gradients.numpy(), labels

        self.predict_fn = predict_fn
        self.paddle_prepared = True
