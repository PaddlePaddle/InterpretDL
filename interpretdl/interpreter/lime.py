import os
import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import paddle

from ..data_processor.readers import preprocess_image, read_image, restore_image
from ..data_processor.visualizer import show_important_parts, visualize_image, save_image
from ..common.paddle_utils import init_checkpoint, to_lodtensor

from ._lime_base import LimeBase
from .abc_interpreter import Interpreter


class LIMECVInterpreter(Interpreter):
    """
    LIME Interpreter for CV tasks.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self,
                 paddle_model: Callable,
                 model_input_shape=[3, 224, 224],
                 use_cuda=True) -> None:
        """
        Initialize the LIMECVInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """

        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.model_input_shape = model_input_shape
        self.use_cuda = use_cuda
        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase()

        self.lime_intermediate_results = {}

    def interpret(self,
                  data,
                  interpret_class=None,
                  num_samples=1000,
                  batch_size=50,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (str): The input file path.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be used. Default: None
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The path to save the processed image. If None, the image will not be saved. Default: None

        :return: LIME Prior weights: {interpret_label_i: weights on features}
        :rtype: dict

        """
        if isinstance(data, str):
            data_instance = read_image(
                data, crop_size=self.model_input_shape[1])
        else:
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0)
            if np.issubdtype(data.dtype, np.integer):
                data_instance = data
            else:
                data_instance = restore_image(data.copy())

        self.input_type = type(data_instance)
        self.data_type = np.array(data_instance).dtype

        if not self.paddle_prepared:
            self._paddle_prepare()
        # only one example here
        probability = self.predict_fn(data_instance)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        interpret_class = np.array(interpret_class)

        lime_weights, r2_scores = self.lime_base.interpret_instance(
            data_instance[0],
            self.predict_fn,
            interpret_class,
            num_samples=num_samples,
            batch_size=batch_size)

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

            def predict_fn(data_instance):
                data = preprocess_image(
                    data_instance
                )  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
                data = paddle.to_tensor(data)
                data.stop_gradient = False
                out = self.paddle_model(data)
                probs = paddle.nn.functional.softmax(out, axis=1)
                return probs.numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True


class LIMENLPInterpreter(Interpreter):
    """
    LIME Interpreter for NLP tasks.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self, paddle_model, use_cuda=True) -> None:
        """
        Initialize the LIMENLPInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            trained_model_path (str): The pretrained model directory.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """

        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.use_cuda = use_cuda
        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase()

        self.lime_intermediate_results = {}

    def interpret(self,
                  data,
                  preprocess_fn,
                  unk_id,
                  pad_id=None,
                  interpret_class=None,
                  num_samples=1000,
                  batch_size=50,
                  lod_levels=None,
                  return_pred=False,
                  visual=True):
        """
        Main function of the interpreter.

        Args:
            data (str): The raw string for analysis.
            preprocess_fn (Callable): A user-defined function that input raw string and outputs the a tuple of inputs to feed into the NLP model.
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. Default: None.
            interpret_class (list or numpy.ndarray, optional): The index of class to interpret. If None, the most likely label will be used. Default: None
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            lod_levels (list or tuple or numpy.ndarray or None, optional): The lod levels for model inputs. It should have the length equal to number of outputs given by preprocess_fn.
                                            If None, lod levels are all zeros. Default: None.
            visual (bool, optional): Whether or not to visualize. Default: True

        :return: LIME Prior weights: {interpret_label_i: weights on features}
        :rtype: dict
        """

        model_inputs = preprocess_fn(data)
        if not isinstance(model_inputs, tuple):
            self.model_inputs = (np.array(model_inputs), )
        else:
            self.model_inputs = tuple(inp.numpy() for inp in model_inputs)

        if not self.paddle_prepared:
            self._paddle_prepare()
        # only one example here
        probability = self.predict_fn(*self.model_inputs)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        lime_weights, r2_scores = self.lime_base.interpret_instance_text(
            self.model_inputs,
            classifier_fn=self.predict_fn,
            interpret_labels=interpret_class,
            unk_id=unk_id,
            pad_id=pad_id,
            num_samples=num_samples,
            batch_size=batch_size)

        data_array = self.model_inputs[0]
        data_array = data_array.reshape((np.prod(data_array.shape), ))
        for c in lime_weights:
            weights_c = lime_weights[c]
            weights_new = [(data_array[tup[0]], tup[1]) for tup in weights_c]
            lime_weights[c] = weights_new

        if return_pred:
            return (interpret_class, probability[interpret_class],
                    lime_weights)
        return lime_weights

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

            def predict_fn(*params):
                params = tuple(paddle.to_tensor(inp) for inp in params)
                probs = self.paddle_model(*params)
                return probs.numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True
