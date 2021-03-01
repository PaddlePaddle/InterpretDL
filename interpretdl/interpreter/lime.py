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
            paddle_model (callable): A user-defined function that gives access to model predictions.
                    It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.
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

        Example::

            import interpretdl as it
            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            lime = it.LIMECVInterpreter(paddle_model, "assets/ResNet50_pretrained")
            lime_weights = lime.interpret(
                    'assets/catdog.png',
                    num_samples=1000,
                    batch_size=100,
                    save_path='assets/catdog_lime.png')

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
            paddle_model (callable): A user-defined function that gives access to model predictions.
                    It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.
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

        Example::

            from assets.bilstm import bilstm
            import io

            from interpretdl.data_processor.visualizer import VisualizationTextRecord, visualize_text

            def load_vocab(file_path):
                vocab = {}
                with io.open(file_path, 'r', encoding='utf8') as f:
                    wid = 0
                    for line in f:
                        if line.strip() not in vocab:
                            vocab[line.strip()] = wid
                            wid += 1
                vocab["<unk>"] = len(vocab)
                return vocab

            MODEL_PATH = "assets/senta_model/bilstm_model"
            VOCAB_PATH = os.path.join(MODEL_PATH, "word_dict.txt")
            PARAMS_PATH = os.path.join(MODEL_PATH, "params")
            DICT_DIM = 1256606

            def paddle_model(data, seq_len):
                probs = bilstm(data, seq_len, None, DICT_DIM, is_prediction=True)
                return probs

            MAX_SEQ_LEN = 256

            def preprocess_fn(data):
                word_ids = []
                sub_word_ids = [word_dict.get(d, unk_id) for d in data.split()]
                seq_lens = [len(sub_word_ids)]
                if len(sub_word_ids) < MAX_SEQ_LEN:
                    sub_word_ids += [0] * (MAX_SEQ_LEN - len(sub_word_ids))
                word_ids.append(sub_word_ids[:MAX_SEQ_LEN])
                return word_ids, seq_lens

            #https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
            word_dict = load_vocab(VOCAB_PATH)
            unk_id = word_dict[""]  #word_dict["<unk>"]
            lime = it.LIMENLPInterpreter(paddle_model, PARAMS_PATH)

            reviews = [
                '交通 方便 ；环境 很好 ；服务态度 很好 房间 较小',
                '这本书 实在 太烂 了 , 什么 朗读 手册 , 一点 朗读 的 内容 都 没有 . 看 了 几页 就 不 想 看 下去 了 .'
            ]

            true_labels = [1, 0]
            recs = []
            for i, review in enumerate(reviews):

                pred_class, pred_prob, lime_weights = lime.interpret(
                    review,
                    preprocess_fn,
                    num_samples=200,
                    batch_size=10,
                    unk_id=unk_id,
                    pad_id=0,
                    return_pred=True)

                id2word = dict(zip(word_dict.values(), word_dict.keys()))
                for y in lime_weights:
                    print([(id2word[t[0]], t[1]) for t in lime_weights[y]])

                words = review.split()
                interp_class = list(lime_weights.keys())[0]
                word_importances = [t[1] for t in lime_weights[interp_class]]
                word_importances = np.array(word_importances) / np.linalg.norm(
                    word_importances)
                true_label = true_labels[i]
                if interp_class == 0:
                    word_importances = -word_importances
                rec = VisualizationTextRecord(words, word_importances, true_label,
                                              pred_class[0], pred_prob[0],
                                              interp_class)
                recs.append(rec)

            visualize_text(recs)
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
