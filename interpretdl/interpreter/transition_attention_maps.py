
import numpy as np
import re

from .abc_interpreter import Interpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class TAMInterpreter(Interpreter):
    """
    Transition Attention Maps Interpreter.

    More details regarding the Transition_Attention_Maps method can be found in the original paper:
    
    .. code-block:: none

        Tingyi Yuan, Xuhong Li, Haoyi Xiong, Hui Cao, Dejing Dou. Explaining Information Flow Inside Vision Transformers Using Markov Chain. 
        In *Neurips 2021 XAI4Debugging Workshop*. https://openreview.net/forum?id=TT-cf6QSDaQ

    """

    def __init__(self,
                 paddle_model,
                 use_cuda=None,
                 device='gpu:0') -> None:
        """

        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu:0``, ``gpu:1`` etc.
            use_cuda (bool):  Would be deprecated soon. Use ``device`` directly.
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda)
        self.paddle_prepared = False

    def interpret(self,
                  inputs,
                  start_layer=4,
                  steps=20,
                  label=None,
                  resize_to=224, 
                  crop_to=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            start_layer (int, optional): Compute the state from the start layer. Default: 4
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. 
                If None, the most likely label for each image will be used. Default: None
            resize_to (int, optional): [description]. Images will be rescaled with the shorter edge being `resize_to`. Defaults to 224.
            crop_to ([type], optional): [description]. After resize, images will be center cropped to a square image with the size `crop_to`. 
                If None, no crop will be performed. Defaults to None.
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        Returns:
            [numpy.ndarray]: interpretations/heatmap for images
        """

        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)
        bsz = len(data)  # batch size
        save_path = preprocess_save_path(save_path, bsz)

        if not self.paddle_prepared:
            self._paddle_prepare()

        attns, _, preds = self.predict_fn(data)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        num_blocks = len(attns)
        states = np.mean(attns[-1], axis=1)[:, 0, :].reshape((b, 1, s))
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = np.mean(attns[i], 1)
            states_ = states
            states = states @ attn
            states += states_

        total_gradients = np.zeros((b, h, s, s))
        for alpha in np.linspace(0, 1, steps):
            # forward propagation
            data_scaled = data * alpha
            _, gradients, _ = self.predict_fn(data_scaled, label=label)

            total_gradients += gradients

        W_state = np.mean((total_gradients / steps).clip(min=0), axis=1)[:, 0, :].reshape((b, 1, s))

        tam_explanation = (states * W_state)[:, 0, 1:].reshape((-1, 14, 14))

        # visualization and save image.
        for i in range(bsz):
            vis_explanation = explanation_to_vis(imgs[i], tam_explanation[i], style='overlay_heatmap')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return tam_explanation

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle
            paddle.set_device(self.device)
            # to get gradients, the ``train`` mode must be set.
            # we cannot set v.training = False for the same reason.
            self.paddle_model.train()

            for n, v in self.paddle_model.named_sublayers():
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0

                # Report issues or pull requests if more layers need to be added.

            def predict_fn(data, label=None):
                data = paddle.to_tensor(data)
                data.stop_gradient = False

                attns = []
                def hook(layer, input, output):
                    attns.append(output)

                hooks = []
                for n, v in self.paddle_model.named_sublayers():
                    if re.match('^blocks.*.attn.attn_drop$', n):
                        h = v.register_forward_post_hook(hook)
                        hooks.append(h)
                out = self.paddle_model(data)
                for h in hooks:
                    h.remove()

                out = paddle.nn.functional.softmax(out, axis=1)
                preds = paddle.argmax(out, axis=1)
                if label is None:
                    label = preds.numpy()

                label_onehot = paddle.nn.functional.one_hot(
                    paddle.to_tensor(label), num_classes=out.shape[1])
                target = paddle.sum(out * label_onehot, axis=1)
                target.backward()
                gradients = attns[-1].grad
                target.clear_gradient()
                if isinstance(gradients, paddle.Tensor):
                    gradients = gradients.numpy()

                a = []
                for attn in attns:
                    a.append(attn.numpy())

                return a, gradients, label

        self.predict_fn = predict_fn
        self.paddle_prepared = True
