import typing
from typing import Any, Callable, List, Tuple, Union

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image, preprocess_inputs
from ..data_processor.visualizer import visualize_grayscale

import numpy as np
import paddle.fluid as fluid
import os, sys
import paddle


class GradShapCVInterpreter(Interpreter):
    """
    Gradient SHAP Interpreter for CV tasks.

    More details regarding the GradShap method can be found in the original paper:
    http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradShapCVInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data inputs.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            class_num (int): Number of classes for the model.
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
                  n_samples=5,
                  noise_amount=0.1,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None            baseline (numpy.ndarray, optional): The baseline input. If None, all zeros will be used. Default: None
            baselines (numpy.ndarray or None, optional): The baseline images to compare with. It should have the same shape as images and same length as the number of images.
                                                        If None, the baselines of all zeros will be used. Default: None.
            n_samples (int, optional): The number of randomly generated samples. Default: 5.
            noise_amount (float, optional): Noise level of added noise to each image.
                                            The std of Guassian random noise is noise_amount * (x_max - x_min). Default: 0.1
            visual (bool, optional): Whether or not to visualize the processed image. Default: True.
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations for images
        :rtype: numpy.ndarray

        Example::

            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            gs = GradShapInterpreter(predict_fn, "assets/ResNet50_pretrained", 1000, True)
            avg_interpretations = gs.interpret(
                                img_path,
                                label=None,
                                noise_amount=0.1,
                                n_samples=5,
                                visual=True,
                                save_path='grad_shap_test.jpg')
        """

        def add_noise_to_inputs(data):
            max_axis = tuple(np.arange(1, data.ndim))
            stds = noise_amount * (
                np.max(data, axis=max_axis) - np.min(data, axis=max_axis))
            noise = np.concatenate([
                np.random.normal(0.0, stds[j], (n_samples, ) + tuple(d.shape))
                for j, d in enumerate(data)
            ]).astype(self.data_type)
            repeated_data = np.repeat(data, (n_samples, ) * len(data), axis=0)
            return repeated_data + noise

        _, data, save_path = preprocess_inputs(inputs, save_path,
                                               self.model_input_shape)

        self.data_type = np.array(data).dtype

        data_with_noise = add_noise_to_inputs(data)
        bsz = len(data)
        if baselines is None:
            baselines = np.zeros_like(data)
        baselines = np.repeat(baselines, (n_samples, ) * bsz, axis=0)

        if not self.paddle_prepared:
            self._paddle_prepare()

        if labels is None:
            _, preds = self.predict_fn(data, labels)
            labels = preds

        labels = np.array(labels).reshape(
            (bsz, 1))  #.repeat(n_samples, axis=0)
        labels = np.repeat(labels, (n_samples, ) * bsz, axis=0)

        rand_scales = np.random.uniform(
            0.0, 1.0, (bsz * n_samples, 1)).astype(self.data_type)

        input_baseline_points = np.array([
            d * r + b * (1 - r)
            for d, r, b in zip(data_with_noise, rand_scales, baselines)
        ])

        gradients, _ = self.predict_fn(input_baseline_points, labels)

        input_baseline_diff = data_with_noise - baselines

        interpretations = gradients * input_baseline_diff

        interpretations = np.concatenate([
            np.mean(
                interpretations[i * n_samples:(i + 1) * n_samples],
                axis=0,
                keepdims=True) for i in range(bsz)
        ])

        for i in range(bsz):
            visualize_grayscale(
                interpretations[i], visual=visual, save_path=save_path[i])

        return interpretations

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


class GradShapNLPInterpreter(Interpreter):
    """
    Gradient SHAP Interpreter for NLP tasks.

    More details regarding the GradShap method can be found in the original paper:
    http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
    """

    def __init__(self, paddle_model, use_cuda=True) -> None:
        """
        Initialize the GradShapNLPInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data input.
                - alpha: A scalar for calculating the path integral
                - baseline: The baseline input.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            class_num (int): Number of classes for the model.
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
                  n_samples=5,
                  noise_amount=0.1,
                  return_pred=False,
                  visual=True):
        """
        Main function of the interpreter.

        Args:
            data (fluid.LoDTensor): The word ids inputs.
            label (list or numpy.ndarray, optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            n_samples (int, optional): The number of randomly generated samples. Default: 5.
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

            def paddle_model(data, alpha, std):
                dict_dim = 1256606
                emb_dim = 128
                # embedding layer
                emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
                #emb += noise
                emb += fluid.layers.gaussian_random(fluid.layers.shape(emb), std=std)
                emb *= alpha
                probs = bilstm_net_emb(emb, None, None, dict_dim, is_prediction=True)
                return emb, probs

            MODEL_PATH = "assets/senta_model/bilstm_model/"
            PARAMS_PATH = os.path.join(MODEL_PATH, "params")
            VOCAB_PATH = os.path.join(MODEL_PATH, "word_dict.txt")

            gs = it.GradShapNLPInterpreter(paddle_model, PARAMS_PATH, True)

            word_dict = load_vocab(VOCAB_PATH)
            unk_id = word_dict["<unk>"]
            reviews = [
                ['交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'],
                ['交通', '不方便', '环境', '很差', '；', '服务态度', '一般', '', '房间', '较小']
            ]

            lod = []
            for c in reviews:
                lod.append([word_dict.get(words, unk_id) for words in c])
            print(lod)
            base_shape = [[len(c) for c in lod]]
            lod = np.array(sum(lod, []), dtype=np.int64)
            data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())

            pred_labels, pred_probs, avg_gradients = gs.interpret(
                data,
                label=None,
                noise_amount=0.1,
                n_samples=20,
                return_pred=True,
                visual=True)

            sum_gradients = np.sum(avg_gradients, axis=1).tolist()
            lod = data.lod()

            new_array = []
            for i in range(len(lod[0]) - 1):
                new_array.append(
                    list(zip(reviews[i], sum_gradients[lod[0][i]:lod[0][i + 1]])))

            print(new_array)
            true_labels = [1, 0]
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
                recs.append(
                    VisualizationTextRecord(words, word_importances, true_label,
                                            pred_label, pred_prob, interp_class))

            visualize_text(recs)
        """

        self.noise_amount = noise_amount
        if not self.paddle_prepared:
            self._paddle_prepare()

        if isinstance(data, tuple):
            n = data[0].shape[0]
        else:
            n = data.shape[0]
        global alpha
        alpha = 1
        gradients, out, embedding = self.predict_fn(data, [0] * n)

        if labels is None:
            labels = np.argmax(out, axis=1)
        else:
            labels = np.array(labels)

        labels = labels.reshape((n, ))

        rand_scales = np.random.uniform(0.0, 1.0, (n_samples, ))

        total_gradients = np.zeros_like(gradients)
        for alpha in rand_scales:
            gradients, _, _ = self.predict_fn(data, labels)
            total_gradients += np.array(gradients)

        avg_gradients = total_gradients / n_samples
        interpretations = avg_gradients * embedding

        if return_pred:
            pred_labels = labels.reshape((n, ))
            pred_probs = [
                p[pred_labels[i]] for i, p in enumerate(np.array(out))
            ]
            return (pred_labels, pred_probs, interpretations)

        return interpretations

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
                output += paddle.normal(
                    std=self.noise_amount, shape=output.shape)
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
