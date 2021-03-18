import typing
from typing import Any, Callable, List, Tuple, Union

import IPython.display as display
import cv2
import numpy as np
import os, sys
from PIL import Image
import paddle
from tqdm import tqdm

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import visualize_heatmap


class ScoreCAMInterpreter(Interpreter):
    """
    Score CAM Interpreter.

    More details regarding the Score CAM method can be found in the original paper:
    https://arxiv.org/abs/1910.01279
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradCAMInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
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
        """

        imgs, data = preprocess_inputs(inputs, self.model_input_shape)

        bsz = len(data)
        save_path = preprocess_save_path(save_path, bsz)

        b, c, h, w = data.shape

        self.target_layer_name = target_layer_name

        if not self.paddle_prepared:
            self._paddle_prepare()

        if labels is None:
            _, probs = self.predict_fn(data)
            labels = np.argmax(probs, axis=1)

        labels = np.array(labels).reshape((bsz, 1))
        feature_map, _ = self.predict_fn(data)
        interpretations = np.zeros((b, h, w))

        for i in tqdm(range(feature_map.shape[1])):
            feature_channel = feature_map[:, i, :, :]
            feature_channel = np.concatenate([
                np.expand_dims(cv2.resize(f, (h, w)), 0)
                for f in feature_channel
            ])
            norm_feature_channel = np.array(
                [(f - f.min()) / (f.max() - f.min())
                 for f in feature_channel]).reshape((b, 1, h, w))
            _, probs = self.predict_fn(data * norm_feature_channel)
            scores = [p[labels[i]] for i, p in enumerate(probs)]
            interpretations += feature_channel * np.array(scores).reshape((
                b, ) + (1, ) * (interpretations.ndim - 1))

        interpretations = np.maximum(interpretations, 0)
        interpretations_min, interpretations_max = interpretations.min(
        ), interpretations.max()

        if interpretations_min == interpretations_max:
            return None

        interpretations = (interpretations - interpretations_min) / (
            interpretations_max - interpretations_min)

        interpretations = np.array([(interp - interp.min()) /
                                    (interp.max() - interp.min())
                                    for interp in interpretations])

        for i in range(b):
            visualize_heatmap(interpretations[i], imgs[i], visual,
                              save_path[i])

        return interpretations

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            if self.use_cuda:
                paddle.set_device('gpu:0')
            else:
                paddle.set_device('cpu')

            self.paddle_model.eval()

            self._feature_maps = None

            def hook(layer, input, output):
                #                 global feature_maps
                self._feature_maps = output

            registered = False
            for n, v in self.paddle_model.named_sublayers():
                if n == self.target_layer_name:
                    v.register_forward_post_hook(hook)
                    registered = True

            assert registered, f"target_layer_name {self.target_layer_name} does not exist in the given model, \
                            please check all valid layer names by [n for n, v in paddle_model.named_sublayers()]"

            def predict_fn(data):
                global feature_maps
                data = paddle.to_tensor(data)
                data.stop_gradient = False
                out = self.paddle_model(data)
                probs = paddle.nn.functional.softmax(out, axis=1)
                return self._feature_maps.numpy(), probs.numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True
