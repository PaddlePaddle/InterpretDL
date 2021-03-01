import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import os, sys
import paddle

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image, preprocess_inputs
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
            paddle_model: A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data input.
                and outputs predictions. See the example at the end of ``interpret()``.22222
            trained_model_path (str): The pretrained model directory.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 244, 244]

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

        Example::

            import interpretdl as it
            def paddle_model(data):
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=data, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs

            img_path = 'assets/fireboat.png'

            #https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification
            ig = it.IntGradCVInterpreter(paddle_model, "assets/ResNet50_pretrained",
                                         True)
            gradients = ig.interpret(
                img_path,
                labels=None,
                baselines='random',
                steps=50,
                num_random_trials=2,
                visual=True,
                save_path='assets/ig_test.jpg')
        """

        self.labels = labels

        if baselines is None:
            num_random_trials = 1

        imgs, data, save_path = preprocess_inputs(inputs, save_path,
                                                  self.model_input_shape)

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
            paddle_model: A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data input.
                - alpha: A scalar for calculating the path integral
                - baseline: The baseline input.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 244, 244]

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
            data (fluid.LoDTensor): The word ids input.
            label (list or numpy.ndarray, optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            noise_amount (float, optional): Noise level of added noise to the embeddings.
                                            The std of Guassian random noise is noise_amount * (x_max - x_min). Default: 0.1
            return_pred (bool, optional): Whether or not to return predicted labels and probabilities. If True, a tuple of predicted labels, probabilities, and interpretations will be returned.
                                        There are useful for visualization. Else, only interpretations will be returned. Default: False.
            visual (bool, optional): Whether or not to visualize. Default: True.

        :return: interpretations for each word or a tuple of predicted labels, probabilities, and interpretations
        :rtype: numpy.ndarray or tuple

        Example::

            import interpretdl as it
            import io
            #Dataset: https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
            #Pretrained Model: https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz
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

            def paddle_model(data, alpha):
                dict_dim = 1256606
                emb_dim = 128
                # embedding layer
                emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
                emb *= alpha
                probs = bilstm_net_emb(emb, None, None, dict_dim, is_prediction=True)
                return emb, probs

            MODEL_PATH = "assets/senta_model/bilstm_model/"
            PARAMS_PATH = os.path.join(MODEL_PATH, "params")
            VOCAB_PATH = os.path.join(MODEL_PATH, "word_dict.txt")

            ig = it.IntGradNLPInterpreter(paddle_model, PARAMS_PATH, True)

            word_dict = load_vocab(VOCAB_PATH)
            unk_id = word_dict["<unk>"]
            reviews = [[
                '交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'
            ]]

            lod = []
            for c in reviews:
                lod.append([word_dict.get(words, unk_id) for words in c])
            base_shape = [[len(c) for c in lod]]
            lod = np.array(sum(lod, []), dtype=np.int64)
            data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())

            pred_labels, pred_probs, avg_gradients = ig.interpret(
                data, label=None, steps=50, return_pred=True, visual=True)

            sum_gradients = np.sum(avg_gradients, axis=1).tolist()
            lod = data.lod()

            new_array = []
            for i in range(len(lod[0]) - 1):
                new_array.append(
                    list(zip(reviews[i], sum_gradients[lod[0][i]:lod[0][i + 1]])))

            print(new_array)

            true_labels = [1]
            recs = []
            for i, l in enumerate(new_array):
                words = [t[0] for t in l]
                word_importances = [t[1] for t in l]
                word_importances = np.array(word_importances) / np.linalg.norm(
                    word_importances)
                pred_label = pred_labels[i]
                pred_prob = pred_probs[i]
                true_label = true_labels[0]
                interp_class = pred_label
                if interp_class == 0:
                    word_importances = -word_importances
                print(words, word_importances)
                recs.append(
                    VisualizationTextRecord(words, word_importances, true_label,
                                            pred_label, pred_prob, interp_class))

            visualize_text(recs)
        """

        if not self.paddle_prepared:
            self._paddle_prepare()

        if isinstance(data, tuple):
            n = data[0].shape[0]
        else:
            n = data.shape[0]

        global alpha
        alpha = 1
        gradients, out, data_out = self.predict_fn(data, [0] * n)

        if labels is None:
            labels = np.argmax(out, axis=1)
        else:
            labels = np.array(labels)

        labels = labels.reshape((n, ))
        total_gradients = np.zeros_like(gradients)
        for alpha in np.linspace(0, 1, steps):
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

            global embedding
            embedding = None

            def hook(layer, input, output):
                global embedding
                global alpha
                output = alpha * output
                embedding = output
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
                    logits = self.paddle_model(*data)
                    probs = paddle.nn.functional.softmax(logits, axis=1)
                else:
                    logits = self.paddle_model(data)
                    probs = paddle.nn.functional.softmax(logits, axis=1)
                labels_onehot = paddle.nn.functional.one_hot(
                    paddle.to_tensor(labels), num_classes=probs.shape[1])
                target = paddle.sum(probs * labels_onehot, axis=1)
                gradients = paddle.grad(
                    outputs=[target], inputs=[embedding])[0]
                return gradients.numpy(), probs.numpy(), embedding.numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True
