from .abc_interpreter import Interpreter, InputGradientInterpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image

import numpy as np
import paddle


class GradShapCVInterpreter(InputGradientInterpreter):
    """
    Gradient SHAP Interpreter for CV tasks.

    More details regarding the GradShap method can be found in the original paper:
    http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=None,
                 device='gpu:0',
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradShapCVInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        InputGradientInterpreter.__init__(self, paddle_model, device, use_cuda)
        self.model_input_shape = model_input_shape

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
        """

        imgs, data = preprocess_inputs(inputs, self.model_input_shape)
        bsz = len(data)
        self.data_type = np.array(data).dtype

        self._build_predict_fn(gradient_of='probability')

        if labels is None:
            _, labels = self.predict_fn(data, labels)

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
        
        data_with_noise = add_noise_to_inputs(data)

        if baselines is None:
            baselines = np.zeros_like(data)
        baselines = np.repeat(baselines, (n_samples, ) * bsz, axis=0)

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
        explanations = gradients * input_baseline_diff

        explanations = np.concatenate([
            np.mean(
                explanations[i * n_samples:(i + 1) * n_samples],
                axis=0,
                keepdims=True) for i in range(bsz)
        ])

        # visualization and save image.
        save_path = preprocess_save_path(save_path, bsz)
        for i in range(bsz):
            vis_explanation = explanation_to_vis(imgs[i], np.abs(explanations[i]).sum(0), style='overlay_grayscale')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return explanations


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
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self, paddle_model, 'gpu:0', use_cuda)
        
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
            data (tuple or paddle.tensor): The inputs to the NLP model.
            label (list or numpy.ndarray, optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            n_samples (int, optional): The number of randomly generated samples. Default: 5.
            noise_amount (float, optional): Noise level of added noise to the embeddings.
                                            The std of Guassian random noise is noise_amount * (x_max - x_min). Default: 0.1
            return_pred (bool, optional): Whether or not to return predicted labels and probabilities. If True, a tuple of predicted labels, probabilities, and interpretations will be returned.
                                        There are useful for visualization. Else, only interpretations will be returned. Default: False.
            visual (bool, optional): Whether or not to visualize. Default: True.

        :return: interpretations for each word or a tuple of predicted labels, probabilities, and interpretations
        :rtype: numpy.ndarray or tuple
        """

        self.noise_amount = noise_amount
        if not self.paddle_prepared:
            self._paddle_prepare()

        if isinstance(data, tuple):
            n = data[0].shape[0]
        else:
            n = data.shape[0]

        self._alpha = 1
        gradients, out, embedding = self.predict_fn(data, [0] * n)
        out = paddle.nn.functional.softmax(paddle.to_tensor(out)).numpy()
        
        if labels is None:
            labels = np.argmax(out, axis=1)
        else:
            labels = np.array(labels)

        labels = labels.reshape((n, ))

        rand_scales = np.random.uniform(0.0, 1.0, (n_samples, ))

        total_gradients = np.zeros_like(gradients)
        for alpha in rand_scales:
            self._alpha = alpha
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

            self._embedding = None

            def hook(layer, input, output):
                output += paddle.normal(
                    std=self.noise_amount, shape=output.shape)
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
                if isinstance(data, tuple):
                    probs = self.paddle_model(*data)
                else:
                    probs = self.paddle_model(data)
                labels_onehot = paddle.nn.functional.one_hot(
                    paddle.to_tensor(labels), num_classes=probs.shape[1])
                target = paddle.sum(probs * labels_onehot, axis=1)
                target.backward()
                gradients = self._embedding.grad
                if isinstance(gradients, paddle.Tensor):
                    gradients = gradients.numpy()
                return gradients, probs.numpy(), self._embedding.numpy(
                )

        self.predict_fn = predict_fn
        self.paddle_prepared = True
