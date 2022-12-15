import re
import numpy as np
from collections.abc import Iterable

try:
    from .abc_interpreter_m import TransformerInterpreter
except:
    from .abc_interpreter import TransformerInterpreter

from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class BTCVInterpreter(TransformerInterpreter):
    """
    Bidirectional Transformer Interpreter.

    This is a specific interpreter for Transformers models, with two sub-processes: attentional perception, reasoning feedback.
    
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
                  ap_mode: str = "head",
                  start_layer: int = 3,
                  steps: int = 20,
                  attn_map_name='^blocks.[0-9]*.attn.attn_drop$', 
                  attn_v_name='^blocks.[0-9]*.attn.qkv$',
                  attn_proj_name='^blocks.[0-9]*.attn.proj$', 
                  gradient_of: str = 'probability',
                  label: int or None = None,
                  resize_to: int = 224,
                  crop_to: int or None = None,
                  visual: bool = True,
                  save_path: str or None = None):
        """
        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy 
                array of read images.
            ap_mode (str, optional): The approximation method of attentioanl perception stage,
                "head" for head-wise, "token" for token-wise. Default: ``head``.
            start_layer (int, optional): Compute the state from the start layer. Default: ``4``.
            steps (int, optional): number of steps in the Riemann approximation of the integral. Default: ``20``.
            attn_map_name (str, optional): The layer name to obtain the attention weights, head-wise/token-wise.
                Default: ``^blocks.*.attn.attn_drop$``.
            attn_v_name (str, optional): The layer name for query, key, value, token-wise.
                Default: ``blocks.*.attn.qkv``.
            attn_proj_name (str, optional): The layer name for linear projection, token-wise.
                Default: ``blocks.*.attn.proj``.
            gradient_of (str, optional): compute the gradient of ['probability', 'logit' or 'loss']. Default: 
                ``'probability'``. Multi-class classification uses probabitliy, while binary classification uses logit.
            label (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels
                should be equal to the number of images. If None, the most likely label for each image will be used. 
                Default: ``None``.
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
        self._build_predict_fn(
            attn_map_name=attn_map_name, 
            attn_v_name=attn_v_name, 
            attn_proj_name=attn_proj_name,
            gradient_of=gradient_of
        )
        
        attns, grads, inputs, values, projs, proba, preds = self.predict_fn(data)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        R = np.eye(s, s, dtype=attns[0].dtype)
        R = np.expand_dims(R, 0)
        
        if ap_mode == 'head':
            for i, attn in enumerate(attns):
                if i < start_layer:
                    continue
                grad = grads[i]
                attn = attn.reshape((-1, attn.shape[-1], attn.shape[-1]))
                grad = grad.reshape((-1, grad.shape[-1], grad.shape[-1]))

                Ih = np.mean(np.abs(np.matmul(np.transpose(attn, [0, 2, 1]), grad)), axis=(-1,-2))
                Ih = Ih.reshape([b, h])/np.sum(Ih, axis=-1)
                attn = np.matmul(Ih.reshape([b,1,h]),attn.reshape([b,h,-1])).reshape([b,s,s])

                R = R + np.matmul(attn, R)
        elif ap_mode == 'token':
            for i, attn in enumerate(attns):
                if i < start_layer:
                    continue
                z = inputs[i]  # [s, 768]
                v = values[i]  # [s, 768] v = ZW_v
                proj = projs[i]  # W_proj
                vproj = np.matmul(v, proj)  # vproj = ZW = ZW_vW_proj
                order = np.linalg.norm(vproj, axis=-1).squeeze()/np.linalg.norm(z, axis=-1).squeeze()
                m = np.diag(order)
                attn = attn.reshape([-1, attn.shape[-2], attn.shape[-1]]).mean(0)  # over heads
                R = R + np.matmul(np.matmul(attn, m), R)
        else:
            assert "please specify the attentional perception mode"
        
        total_gradients = np.zeros((b, h, s, s))
        for alpha in np.linspace(0, 1, steps):
            # forward propagation
            data_scaled = data * alpha
            _, gradients, _, _, _, _, _ = self.predict_fn(data_scaled, label)

            total_gradients += gradients[-1]

        # gradient mean over heads.
        grad_head_mean = np.mean((total_gradients / steps).clip(min=0), axis=1)  # [b, s, s]

        if (not hasattr(self.model, 'global_pool')) or (self.model.global_pool == 'token'):
            explanation = R[:, 0, :] * grad_head_mean[:, 0, :]
        else:
            # For those that use globa_pooling, e.g., MAE ViT.
            explanation = (R * grad_head_mean)[:, 1:, :].mean(axis=1)

        explanation = explanation[:, 1:].reshape((-1, 14, 14))

        # intermediate results, for possible further usages.
        self.predicted_label = preds
        self.predicted_proba = proba
        self.ap = R
        self.rf = grad_head_mean

        # visualization and save image.
        vis_explanation = explanation_to_vis(imgs, explanation[0], style='overlay_heatmap')
        if visual:
            show_vis_explanation(vis_explanation)
        if save_path is not None:
            save_image(save_path, vis_explanation)

        return explanation


class BTNLPInterpreter(TransformerInterpreter):
    """
    Bidirectional Transformer Interpreter.

    This is a specific interpreter for Transformers models, with two sub-processes: attentional perception, reasoning feedback.
    
    The following implementation is specially designed for Ernie.
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
                  label: list or np.ndarray = None,                  
                  ap_mode: str = "head",
                  start_layer: int = 11,
                  steps: int = 20,
                  embedding_name='^[a-z]*.embeddings$', 
                  attn_map_name='^[a-z]*.encoder.layers.[0-9]*.self_attn.attn_drop$', 
                  attn_v_name='^[a-z]*.encoder.layers.[0-9]*.self_attn.v_proj$',
                  attn_proj_name='^[a-z]*.encoder.layers.[0-9]*.self_attn.out_proj$',
                  gradient_of: str = 'logit',
                  max_seq_len=128,
                  visual=False):
        """
        Args:
            data (str or list of strs or numpy.ndarray): The input text filepath or a list of filepaths or numpy
                array of read texts.
            ap_mode (str, default to head-wise): The approximation method of attentioanl perception stage,
                "head" for head-wise, "token" for token-wise. Default: ``head``.
            start_layer (int, optional): Compute the state from the start layer. Default: ``11``.
            steps (int, optional): number of steps in the Riemann approximation of the integral. Default: ``20``.
            embedding_name (str, optional): The layer name for embedding, head-wise/token-wise.
                Default: ``^ernie.embeddings$``.
            attn_map_name (str, optional): The layer name to obtain the attention weights, head-wise/token-wise.
                Default: ``^ernie.encoder.layers.*.self_attn.attn_drop$``.
            attn_v_name (str, optional): The layer name for value projection, token-wise.
                Default: ``^ernie.encoder.layers.*.self_attn.v_proj$``.
            attn_proj_name (str, optional): The layer name for linear projection, token-wise.
                Default: ``ernie.encoder.layers.*.self_attn.out_proj$``.
            gradient_of (str, optional): compute the gradient of ['probability', 'logit' or 'loss']. Default: 
                ``'logit'``. Multi-class classification uses probabitliy, while binary classification uses logit.
            label (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels
                should be equal to the number of texts. If None, the most likely label for each text will be used.
                Default: ``None``.

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

        self._build_predict_fn(
            embedding_name=embedding_name, 
            attn_map_name=attn_map_name, 
            attn_v_name=attn_v_name, 
            attn_proj_name=attn_proj_name, 
            gradient_of=gradient_of)
        
        attns, grads, inputs, values, projs, proba, preds = self.predict_fn(model_input)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        R = np.eye(s, s, dtype=attns[0].dtype)
        R = np.expand_dims(R, 0)
        
        if ap_mode == 'head':
            for i, attn in enumerate(attns):
                if i < start_layer:
                    continue
                grad = grads[i]
                attn = attn.reshape((-1, attn.shape[-1], attn.shape[-1]))
                grad = grad.reshape((-1, grad.shape[-1], grad.shape[-1]))

                Ih = np.mean(np.abs(np.matmul(np.transpose(attn, [0, 2, 1]), grad)), axis=(-1,-2))
                Ih = Ih.reshape([b, h])/np.sum(Ih, axis=-1)
                attn = np.matmul(Ih.reshape([b,1,h]),attn.reshape([b,h,-1])).reshape([b,s,s])

                R = R + np.matmul(attn, R)
        elif ap_mode == 'token':
            for i, attn in enumerate(attns):
                if i < start_layer:
                    continue
                # attn: [1, 12, 512, 512]
                z = inputs[i]  # [1, 512, 768]
                v = values[i]  # [512, 768] v = ZW_v
                proj = projs[i]  # [768, 768] W_proj
                vproj = np.matmul(v, proj)  # vproj = ZW = ZW_vW_proj
                order = np.linalg.norm(vproj, axis=-1).squeeze()/np.linalg.norm(z, axis=-1).squeeze()
                m = np.diag(order)
                attn = attn[0].mean(0)  # over heads
                R = R + np.matmul(np.matmul(attn, m), R)
        else:
            assert "please specify the attentional perception mode"

        total_gradients = np.zeros((b, h, s, s))
        for alpha in np.linspace(0, 1, steps):
            # forward propagation
            _, gradients, _, _, _, _, _ = self.predict_fn(model_input, label, scale=alpha)
            total_gradients += gradients[-1]

        grad_head_mean = np.mean((total_gradients / steps).clip(min=0), axis=1)  # [b, s, s]
        explanation = R[:, 0, :] * grad_head_mean[:, 0, :]  # NLP tasks return explanations for all tokens, including [CLS] and [SEP].

        # intermediate results, for possible further usages.
        self.predicted_label = preds
        self.predicted_proba = proba
        self.ap = R[:, 0, :]
        self.rf = grad_head_mean[:, 0, :]

        if visual:
            # TODO: visualize if tokenizer is given.
            print("Visualization is not supported yet.")
            print("Currently please see the tutorial for the visualization:")
            print("https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb")

        return explanation