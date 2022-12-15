import re
import numpy as np
from collections.abc import Iterable

try:
    from .abc_interpreter_m import Interpreter, InputGradientInterpreter, TransformerInterpreter
except:
    from .abc_interpreter import Interpreter, InputGradientInterpreter, TransformerInterpreter

from ..data_processor.readers import images_transform_pipeline
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class GAInterpreter(InputGradientInterpreter):
    """
    Generic Attention Interpreter.

    This is a specific interpreter for Bi-Modal Transformers models. GAInterpreter computes the attention map with 
    gradients, and follows the operations that are similar to Rollout, with advanced modifications.

    This implementation is suitable for models with self-attention in each modality, like 
    `CLIP <https://arxiv.org/abs/2103.00020>`_.

    More details regarding the Generic Attention method can be found in the original paper:
    https://arxiv.org/abs/2103.15679.

    """

    def __init__(self, model: callable, device: str = 'gpu:0') -> None:
        """

        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        Interpreter.__init__(self, model, device)

    def interpret(self,
                  image_input: str or np.ndarray,
                  text: str,
                  text_tokenized: np.ndarray,
                  vis_attn_layer_pattern: str = '^visual.transformer.resblocks.[0-9]*.attn.attn_map$',
                  txt_attn_layer_pattern: str = '^transformer.resblocks.[0-9]*.attn.attn_map$',
                  start_layer: int = 11,
                  start_layer_text: int = 11,
                  resize_to: int = 224,
                  crop_to: int or None = None,
                  visual: bool = True,
                  save_path: str or None = None) -> tuple:
        """
        Given ``image_input`` and ``text_tokenized``, GAInterpreter first obtains all attention maps (of layers whose 
        name matches ``vis_attn_layer_pattern`` or ``txt_attn_layer_pattern`` for visual and text modalities 
        respectively) and their gradients of the prediction. Then, GAInterpreter computes the multiplication between 
        the attention map and its gradient for each block, and obtains the matrix multiplication of all blocks. The 
        ``start_layer`` controls the number of involved layers. The order of involving attention maps (from the first
        layer to the last) is the same as Rollout (from first to last). 

        Args:
            image_input (str or np.ndarray): The input image filepath or a list of filepaths or numpy array of read 
                images.
            text (str): original texts, for visualization.
            text_tokenized (np.ndarray): The tokenized text for the model's input directly.
            vis_attn_layer_pattern (str, optional): the pattern name of the layers whose features will output for 
                visual modality. Defaults to ``'^visual.transformer.resblocks.*.attn.attn_map$'``.
            txt_attn_layer_pattern (str, optional): the pattern name of the layers whose features will output for 
                text modality. Defaults to ``'^transformer.resblocks.*.attn.attn_map$'``.
            start_layer (int, optional): Compute the state from the start layer for visual modality. Defaults to 
                ``11``.
            start_layer_text (int, optional): Compute the state from the start layer for text modality. Defaults to 
                ``11``.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The filepath(s) to save the processed image(s). If None, the image will not be 
                saved. Default: ``None``.

        Returns:
            tuple: (text_relevance: np.ndarray, image_relevance: np.ndarray)
        """

        imgs, data = images_transform_pipeline(image_input, resize_to, crop_to)
        bsz = data.shape[0]
        assert bsz == 1, "Only support one image-text pair as input."

        self._build_predict_fn(vis_attn_layer_pattern=vis_attn_layer_pattern,
                               txt_attn_layer_pattern=txt_attn_layer_pattern)

        img_attns, txt_attns, img_attns_grads, txt_attns_grads = self.predict_fn(data, text_tokenized)

        num_tokens = img_attns[0].shape[-1]
        R = np.eye(num_tokens, num_tokens, dtype=img_attns[0].dtype)
        R = np.expand_dims(R, 0)
        R = R.repeat(bsz, axis=0)

        for i, blk in enumerate(img_attns):
            if i < start_layer:
                continue
            grad = img_attns_grads[i]
            cam = blk
            cam = cam.reshape((-1, cam.shape[-1], cam.shape[-1]))
            grad = grad.reshape((-1, grad.shape[-1], grad.shape[-1]))
            cam = grad * cam
            cam = cam.reshape((bsz, -1, cam.shape[-1], cam.shape[-1]))
            cam = cam.clip(min=0).mean(axis=1)
            R = R + np.matmul(cam, R)
        image_relevance = R[0, 0, 1:]
        image_relevance = image_relevance.reshape((7, 7))  # TODO: token's number.

        num_tokens = txt_attns[0].shape[-1]
        R_text = np.eye(num_tokens, num_tokens, dtype=img_attns[0].dtype)
        R_text = np.expand_dims(R_text, 0)
        R_text = R_text.repeat(bsz, axis=0)

        for i, blk in enumerate(txt_attns):
            if i < start_layer_text:
                continue
            grad = txt_attns_grads[i]
            cam = blk
            cam = cam.reshape((-1, cam.shape[-1], cam.shape[-1]))
            grad = grad.reshape((-1, grad.shape[-1], grad.shape[-1]))
            cam = grad * cam
            cam = cam.reshape((bsz, -1, cam.shape[-1], cam.shape[-1]))
            cam = cam.clip(min=0).mean(axis=1)
            R_text = R_text + np.matmul(cam, R_text)
        text_relevance = R_text

        # visualization and save image.
        vis_explanation = explanation_to_vis(imgs, image_relevance, style='overlay_heatmap')
        if visual:
            show_vis_explanation(vis_explanation)
            # TODO: visualizing the texts.
        if save_path is not None:
            save_image(save_path, vis_explanation)

        return text_relevance, image_relevance

    def _build_predict_fn(self, vis_attn_layer_pattern: str, txt_attn_layer_pattern: str, rebuild: bool = False):
        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        if self.predict_fn is None or rebuild:
            import paddle
            self._env_setup()  # inherit from InputGradientInterpreter

            def predict_fn(image, text_tokenized):
                image = paddle.to_tensor(image)
                text_tokenized = paddle.to_tensor(text_tokenized)
                #################################
                ## hooks to get attention maps ##
                img_attns = []

                def img_hook(layer, input, output):
                    img_attns.append(output)

                txt_attns = []

                def txt_hook(layer, input, output):
                    txt_attns.append(output)

                hooks = []  # for remove.
                for n, v in self.model.named_sublayers():
                    if re.match(vis_attn_layer_pattern, n):
                        h = v.register_forward_post_hook(img_hook)
                        hooks.append(h)

                    if re.match(txt_attn_layer_pattern, n):
                        h = v.register_forward_post_hook(txt_hook)
                        hooks.append(h)

                logits_per_image, logits_per_text = self.model(image, text_tokenized)

                for h in hooks:
                    h.remove()
                ## hooks to get attention maps ##
                #################################

                batch_size = text_tokenized.shape[0]
                index = [i for i in range(batch_size)]
                one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
                one_hot[paddle.arange(logits_per_image.shape[0]), index] = 1
                one_hot = paddle.to_tensor(one_hot)
                one_hot = paddle.sum(one_hot * logits_per_image)
                self.model.clear_gradients()
                one_hot.backward()

                img_attns_grads = []
                for i, blk in enumerate(img_attns):
                    grad = blk.grad
                    if isinstance(grad, paddle.Tensor):
                        grad = grad.numpy()
                    img_attns_grads.append(grad)
                    img_attns[i] = blk.numpy()

                txt_attns_grads = []
                for i, blk in enumerate(txt_attns):
                    grad = blk.grad
                    if isinstance(grad, paddle.Tensor):
                        grad = grad.numpy()
                    txt_attns_grads.append(grad)
                    txt_attns[i] = blk.numpy()

                return img_attns, txt_attns, img_attns_grads, txt_attns_grads

            self.predict_fn = predict_fn

            
class GANLPInterpreter(TransformerInterpreter):
    """
    Generic Attention Interpreter.

    This is a specific interpreter for Bi-Modal Transformers models. GAInterpreter computes the attention map with
    gradients, and follows the operations that are similar to Rollout, with advanced modifications.

    The following implementation is specially designed for Ernie.

    More details regarding the Generic Attention method can be found in the original paper:
    https://arxiv.org/abs/2103.15679.

    """

    def __init__(self, model: callable, device: str = 'gpu:0', **kwargs) -> None:
        """

        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        TransformerInterpreter.__init__(self, model, device, **kwargs)

    def interpret(self,
                  raw_text: str,
                  tokenizer: callable = None,
                  text_to_input_fn: callable = None,
                  label: int or None = None,
                  start_layer: int = 11,
                  attn_map_name='^[a-z]*.encoder.layers.[0-9]*.self_attn.attn_drop$',
                  gradient_of: str = 'logit',
                  max_seq_len=128,
                  visual=False):
        """
        Args:
            data (str or list of strs or numpy.ndarray): The input text filepath or a list of filepaths or numpy
                array of read texts.
            start_layer (int, optional): Compute the state from the start layer. Default: ``11``.
            label (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels
                should be equal to the number of texts. If None, the most likely label for each text will be used.
                Default: ``None``.
            attn_map_name (str, optional): The layer name to obtain attention weights.
                Default: ``^ernie.encoder.layers.*.self_attn.attn_drop$``.
            gradient_of (str, optional): compute the gradient of ['probability', 'logit' or 'loss']. Default: 
                ``'logit'``. Multi-class classification uses probabitliy, while binary classification uses logit.

        Returns:
            [numpy.ndarray]: interpretations for texts
        """
        assert (tokenizer is None) + (text_to_input_fn is None) == 1, "only one of them should be given."

        # tokenizer to text_to_input_fn.
        if tokenizer is not None:
            def text_to_input_fn(raw_text):
                encoded_inputs = tokenizer(text=raw_text, max_seq_len=max_seq_len)
                # order is important. *_batched_and_to_tuple will be the input for the model.
                _batched_and_to_tuple = tuple([np.array([v]) for v in encoded_inputs.values()])
                return _batched_and_to_tuple
        else:
            print("Warning: Visualization can not be supported if tokenizer is not given.")          

        # from raw text string to token ids (and other terms that the user-defined function outputs).
        model_input = text_to_input_fn(raw_text)
        if isinstance(model_input, Iterable) and not hasattr(model_input, 'shape'):
            model_input = tuple(inp for inp in model_input)
        else:
            model_input = tuple(model_input, )
        self._build_predict_fn(attn_map_name=attn_map_name, gradient_of=gradient_of)

        attns, grads, inputs, values, projs, proba, preds = self.predict_fn(model_input)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"
        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        R = np.eye(s, s, dtype=attns[0].dtype)
        R = np.expand_dims(R, 0)
              
        for i, blk in enumerate(attns):
            if i < start_layer:
                continue
            grad = grads[i]
            cam = blk
            cam = cam.reshape((b, h, cam.shape[-1], cam.shape[-1])).mean(1)
            grad = grad.reshape((b, h, grad.shape[-1], grad.shape[-1])).mean(1)
            
            cam = (cam*grad).reshape([b,s,s]).clip(min=0)

            R = R + np.matmul(cam, R)

        explanation = R[:, 0]  # NLP tasks return explanations for all tokens, including [CLS] and [SEP].

        # intermediate results, for possible further usages.
        self.predicted_label = preds
        self.predicted_proba = proba

        if visual:
            # TODO: visualize if tokenizer is given.
            print("Visualization is not supported yet.")
            print("Currently please see the tutorial for the visualization:")
            print("https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb")

        return explanation


class GACVInterpreter(TransformerInterpreter):
    """
    The following implementation is specially designed for Vision Transformer.
    """

    def __init__(self, model: callable, device: str = 'gpu:0', **kwargs) -> None:
        """

        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        TransformerInterpreter.__init__(self, model, device, **kwargs)

    def interpret(self,
                  inputs: str or list(str) or np.ndarray,
                  start_layer: int = 3,
                  attn_map_name='^blocks.[0-9]*.attn.attn_drop$', 
                  label: int or None = None,
                  gradient_of: str = 'probability',
                  resize_to: int = 224,
                  crop_to: int or None = None,
                  visual: bool = True,
                  save_path: str or None = None):
        """
        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy 
                array of read images.
            start_layer (int, optional): Compute the state from the start layer. Default: ``4``.
            attn_map_name (str, optional):  The layer name to obtain attention weights.
                Default: ``^blocks.*.attn.attn_drop$``
            label (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels
                should be equal to the number of images. If None, the most likely label for each image will be used. 
                Default: ``None``.
            gradient_of (str, optional): compute the gradient of ['probability', 'logit' or 'loss']. Default: 
                ``'probability'``. Multi-class classification uses probabitliy, while binary classification uses logit.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The filepath(s) to save the processed image(s). If None, the image will not be 
                saved. Default: ``None``.

        Returns:
            [numpy.ndarray]: interpretations/heatmap for images
        """

        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)
        b = len(data)  # batch size
        assert b==1, "only support single image"
        self._build_predict_fn(attn_map_name=attn_map_name, gradient_of=gradient_of)
        
        attns, grads, inputs, values, projs, proba, preds = self.predict_fn(data, label=label)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        b, h, s, _ = attns[0].shape
        R = np.eye(s, s, dtype=attns[0].dtype)
        R = np.expand_dims(R, 0)
              
        for i, attn in enumerate(attns):
            if i < start_layer:
                continue
            grad = grads[i]
            attn = attn.reshape((b, h, attn.shape[-1], attn.shape[-1])).mean(1)
            grad = grad.reshape((b, h, grad.shape[-1], grad.shape[-1])).mean(1)
            attn = (attn*grad).reshape([b,s,s]).clip(min=0)

            R = R + np.matmul(attn, R)

        if (not hasattr(self.model, 'global_pool')) or (self.model.global_pool == 'token'):
            R = R[:, 0, :]
        else:
            # For those that use globa_pooling, e.g., MAE ViT.
            R = R[:, 1:, :].mean(axis=1)
        explanation = R[:, 1:].reshape((-1, 14, 14))

        # visualization and save image.
        vis_explanation = explanation_to_vis(imgs, explanation[0], style='overlay_heatmap')
        if visual:
            show_vis_explanation(vis_explanation)
        if save_path is not None:
            save_image(save_path, vis_explanation)

        return explanation
