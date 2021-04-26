import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import os, sys
import paddle

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import visualize_overlay


class IntGradCVInterpreter(Interpreter):
    """
    Integrated Gradients Interpreter for CV tasks.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the IntGradCVInterpreter.

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
                  labels=None,
                  baselines=None,
                  steps=50,
                  num_random_trials=10,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None            baseline (numpy.ndarray, optional): The baseline input. If None, all zeros will be used. Default: None
            baselines (numpy.ndarray or None, optional): The baseline images to compare with. It should have the same shape as images and same length as the number of images.
                                                        If None, the baselines of all zeros will be used. Default: None.
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            num_random_trials (int, optional): number of random initializations to take average in the end. Default: 10
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/gradients for images
        :rtype: numpy.ndarray
        """

        self.labels = labels

        if baselines is None:
            num_random_trials = 1

        imgs, data = preprocess_inputs(inputs, self.model_input_shape)

        bsz = len(data)
        save_path = preprocess_save_path(save_path, bsz)

        self.data_type = np.array(data).dtype
        self.input_type = type(data)

        if baselines is None:
            self.baselines = np.zeros(
                (num_random_trials, ) + data.shape, dtype=self.data_type)
        elif baselines == 'random':
            self.baselines = np.random.normal(
                size=(num_random_trials, ) + data.shape).astype(self.data_type)
        else:
            self.baselines = baselines

        if not self.paddle_prepared:
            self._paddle_prepare()

        n = data.shape[0]

        gradients, preds = self.predict_fn(data, labels)

        if self.labels is None:
            self.labels = preds.reshape((n, ))
        else:
            self.labels = np.array(self.labels).reshape((n, ))

        gradients_list = []
        for i in range(num_random_trials):
            total_gradients = np.zeros_like(gradients)
            for alpha in np.linspace(0, 1, steps):
                data_scaled = data * alpha + self.baselines[i] * (1 - alpha)
                gradients, _ = self.predict_fn(data_scaled, self.labels)
                total_gradients += gradients
            ig_gradients = total_gradients * (data - self.baselines[i]) / steps
            gradients_list.append(ig_gradients)
        avg_gradients = np.average(np.array(gradients_list), axis=0)

        for i in range(len(imgs)):
            visualize_overlay(avg_gradients[i], imgs[i], visual, save_path[i])

        return avg_gradients

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            if self.use_cuda:
                paddle.set_device('gpu:0')
            else:
                paddle.set_device('cpu')

            self.paddle_model.train()

            for n, v in self.paddle_model.named_sublayers():
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0

            def predict_fn(data, labels):
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
                gradients = paddle.grad(outputs=[target], inputs=[data])[0]
                return gradients.numpy(), labels

        self.predict_fn = predict_fn
        self.paddle_prepared = True


class IntGradNLPInterpreter(Interpreter):
    """
    Integrated Gradients Interpreter for NLP tasks.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self, paddle_model, use_cuda=True) -> None:
        """
        Initialize the IntGradInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]

        """
        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.use_cuda = use_cuda
        self.paddle_prepared = False

    def interpret(self,
                  data,
                  labels=None,
                  steps=50,
                  return_pred=True,
                  visual=True):
        """
        Main function of the interpreter.

        Args:
            data (tuple or paddle.tensor): The inputs to the NLP model.
            label (list or numpy.ndarray, optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            noise_amount (float, optional): Noise level of added noise to the embeddings.
                                            The std of Guassian random noise is noise_amount * (x_max - x_min). Default: 0.1
            return_pred (bool, optional): Whether or not to return predicted labels and probabilities. If True, a tuple of predicted labels, probabilities, and interpretations will be returned.
                                        There are useful for visualization. Else, only interpretations will be returned. Default: False.
            visual (bool, optional): Whether or not to visualize. Default: True.

        :return: interpretations for each word or a tuple of predicted labels, probabilities, and interpretations
        :rtype: numpy.ndarray or tuple
        """

        if not self.paddle_prepared:
            self._paddle_prepare()

        if isinstance(data, tuple):
            n = data[0].shape[0]
        else:
            n = data.shape[0]

        self._alpha = 1
        gradients, out, data_out = self.predict_fn(data, [0] * n)

        if labels is None:
            labels = np.argmax(out, axis=1)
        else:
            labels = np.array(labels)

        labels = labels.reshape((n, ))
        total_gradients = np.zeros_like(gradients)
        for alpha in np.linspace(0, 1, steps):
            self._alpha = alpha
            gradients, _, emb = self.predict_fn(data, labels)
            total_gradients += gradients

        ig_gradients = total_gradients * data_out / steps

        #avg_gradients = np.average(np.array(gradients_list), axis=0)

        if return_pred:
            out = np.array(out)
            return labels, [o[labels[i]]
                            for i, o in enumerate(out)], ig_gradients

        return ig_gradients

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            if self.use_cuda:
                paddle.set_device('gpu:0')
            else:
                paddle.set_device('cpu')

            self.paddle_model.train()

            self._embedding = None

            def hook(layer, input, output):
                output = self._alpha * output
                self._embedding = output
                return output

            for n, v in self.paddle_model.named_sublayers():
                if "embedding" in v.__class__.__name__.lower():
                    v.register_forward_post_hook(hook)
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0

            def predict_fn(data, labels):
                global embedding
                if isinstance(data, tuple):
                    probs = self.paddle_model(*data)
                else:
                    probs = self.paddle_model(data)
                labels_onehot = paddle.nn.functional.one_hot(
                    paddle.to_tensor(labels), num_classes=probs.shape[1])
                target = paddle.sum(probs * labels_onehot, axis=1)
                gradients = paddle.grad(
                    outputs=[target], inputs=[self._embedding])[0]
                return gradients.numpy(), probs.numpy(), self._embedding.numpy(
                )

        self.predict_fn = predict_fn
        self.paddle_prepared = True
