import numpy as np
from tqdm import tqdm
from .abc_interpreter import InputGradientInterpreter, Interpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class IntGradCVInterpreter(InputGradientInterpreter):
    """
    Integrated Gradients Interpreter for CV tasks.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(
        self,
        paddle_model: callable,
        use_cuda: bool=None,
        device: str='gpu:0'
    ):
        """

        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu:0``, ``gpu:1`` etc.
            use_cuda (bool):  Would be deprecated soon. Use ``device`` directly.
        """
        InputGradientInterpreter.__init__(self, paddle_model, device, use_cuda)

    def interpret(
        self,
        inputs: str or list(str) or np.ndarray,
        labels: list or tuple or np.ndarray or None=None,
        baselines: np.ndarray or None=None,
        steps: int=50,
        num_random_trials: int=10,
        resize_to=224, 
        crop_to=None,
        visual: bool=True,
        save_path=None
    ):
        """Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None            baseline (numpy.ndarray, optional): The baseline input. If None, all zeros will be used. Default: None
            baselines (numpy.ndarray or None, optional): The baseline images to compare with. It should have the same shape as images and same length as the number of images.
                                                        If None, the baselines of all zeros will be used. Default: None.
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            num_random_trials (int, optional): number of random initializations to take average in the end. Default: 10
            resize_to (int, optional): [description]. Images will be rescaled with the shorter edge being `resize_to`. Defaults to 224.
            crop_to ([type], optional): [description]. After resize, images will be center cropped to a square image with the size `crop_to`. 
                If None, no crop will be performed. Defaults to None.
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        Returns:
            [numpy.ndarray]: interpretations/gradients for images
        """

        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)

        self.data_type = np.array(data).dtype
        self.input_type = type(data)

        self._build_predict_fn(gradient_of='probability')

        if baselines is None:
            num_random_trials = 1
            self.baselines = np.zeros(
                (num_random_trials, ) + data.shape, dtype=self.data_type)
        elif baselines == 'random':
            self.baselines = np.random.normal(
                size=(num_random_trials, ) + data.shape).astype(self.data_type)
        else:
            self.baselines = baselines
        bsz = len(data)

        # obtain the labels (and initialization).
        if labels is None:
            gradients, preds = self.predict_fn(data, labels)
            labels = preds
        labels = np.array(labels).reshape((bsz, ))

        # IntGrad.
        gradients_list = []
        with tqdm(total=num_random_trials * steps, leave=True, position=0) as pbar:
            for i in range(num_random_trials):
                total_gradients = np.zeros_like(data)
                for alpha in np.linspace(0, 1, steps):
                    data_scaled = data * alpha + self.baselines[i] * (1 - alpha)
                    gradients, _ = self.predict_fn(data_scaled, labels)
                    total_gradients += gradients
                    pbar.update(1)

                ig_gradients = total_gradients * (data - self.baselines[i]) / steps
                gradients_list.append(ig_gradients)
        avg_gradients = np.average(np.array(gradients_list), axis=0)

        # visualization and save image.
        if save_path is None and not visual:
            # no need to visualize or save explanation results.
            pass
        else:
            save_path = preprocess_save_path(save_path, bsz)
            for i in range(bsz):
                vis_explanation = explanation_to_vis(imgs[i], np.abs(avg_gradients[i]).sum(0), style='overlay_grayscale')
                if visual:
                    show_vis_explanation(vis_explanation)
                if save_path[i] is not None:
                    save_image(save_path[i], vis_explanation)

        # intermediate results, for possible further usages.
        self.labels = labels

        return avg_gradients


class IntGradNLPInterpreter(Interpreter):
    """
    Integrated Gradients Interpreter for NLP tasks.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self, paddle_model, use_cuda=None, device='gpu:0') -> None:
        """

        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu:0``, ``gpu:1`` etc.
            use_cuda (bool):  Would be deprecated soon. Use ``device`` directly.
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda)

    def interpret(self,
                  data,
                  labels=None,
                  steps=50,
                  embedding_name='word_embeddings',
                  return_pred=True):
        """Main function of the interpreter.

        Args:
            data ([type]): [description]
            labels ([type], optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            embedding_name (str, optional): name of the embedding layer at which the steps will be applied. 
                Defaults to 'word_embeddings'. The correct name of embedding can be found through ``print(model)``.
            return_pred (bool, optional): Whether or not to return predicted labels and probabilities. 
                If True, a tuple of predicted labels, probabilities, and interpretations will be returned.
                There are useful for visualization. Else, only interpretations will be returned. Default: True.

        Returns:
            [numpy.ndarray or tuple]: interpretations for each word or a tuple of predicted labels, probabilities, and interpretations.
        """

        self._build_predict_fn(embedding_name=embedding_name, gradient_of='probability')

        if isinstance(data, tuple):
            bs = data[0].shape[0]
        else:
            bs = data.shape[0]

        gradients, labels, data_out, probas = self.predict_fn(data, labels, None)
        labels = labels.reshape((bs, ))
        total_gradients = np.zeros_like(gradients)
        for alpha in np.linspace(0, 1, steps):
            gradients, _, _, _ = self.predict_fn(data, labels, alpha)
            total_gradients += gradients
        
        ig_gradients = total_gradients * data_out / steps
        ig_gradients = np.sum(ig_gradients, axis=-1)

        if return_pred:
            return labels, probas.numpy(), ig_gradients
        
        # Visualization is currently not supported here.
        # See the tutorial for more information:
        # https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2-tutorials.ipynb

        return ig_gradients

    def _build_predict_fn(self, rebuild=False, embedding_name='word_embeddings', gradient_of='probability'):
        
        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."
            return
        
        import paddle
        if self.predict_fn is None or rebuild:
            assert gradient_of in ['loss', 'logit', 'probability']

            if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
                print("Paddle is not installed with GPU support. Change to CPU version now.")
                self.device = 'cpu'

            # set device. self.device is one of ['cpu', 'gpu:0', 'gpu:1', ...]
            paddle.set_device(self.device)

            # to get gradients, the ``train`` mode must be set.
            self.paddle_model.train()
            
            # later version will be simplied.
            for n, v in self.paddle_model.named_sublayers():
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0
                    
            def predict_fn(data, labels, noise_scale=1.0):
                if isinstance(data, tuple):
                    # NLP models usually have two inputs.
                    bs = data[0].shape[0]
                    data = (paddle.to_tensor(data[0]), paddle.to_tensor(data[1]))
                else:
                    bs = data.shape[0]
                    data = paddle.to_tensor(data)
                    
                assert labels is None or \
                    (isinstance(labels, (list, np.ndarray)) and len(labels) == bs)
            
                target_feature_map = []
                def hook(layer, input, output):
                    if noise_scale is not None:
                        output = noise_scale * output
                    target_feature_map.append(output)
                    return output
                hooks = []
                for name, v in self.paddle_model.named_sublayers():
                    if embedding_name in name:
                        h = v.register_forward_post_hook(hook)
                        hooks.append(h)
                        
                if isinstance(data, tuple):
                    logits = self.paddle_model(*data)  # get logits, [bs, num_c]
                else:
                    logits = self.paddle_model(data)  # get logits, [bs, num_c]
                    
                for h in hooks:
                    h.remove()
                    
                probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
                preds = paddle.argmax(probas, axis=1)  # get predictions.
                if labels is None:
                    labels = preds.numpy()  # label is an integer.
                
                if gradient_of == 'loss':
                    # loss
                    loss = paddle.nn.functional.cross_entropy(
                        logits, paddle.to_tensor(labels), reduction='sum'
                    )
                else:
                    # logits or probas
                    labels = np.array(labels).reshape((bs, ))
                    labels_onehot = paddle.nn.functional.one_hot(
                        paddle.to_tensor(labels), num_classes=probas.shape[1]
                    )
                    if gradient_of == 'logit':
                        loss = paddle.sum(logits * labels_onehot, axis=1)
                    else:
                        loss = paddle.sum(probas * labels_onehot, axis=1)

                loss.backward()
                gradients = target_feature_map[0].grad  # get gradients of "embedding".
                loss.clear_gradient()

                if isinstance(gradients, paddle.Tensor):
                    gradients = gradients.numpy()
                return gradients, labels, target_feature_map[0].numpy(), probas

        self.predict_fn = predict_fn
