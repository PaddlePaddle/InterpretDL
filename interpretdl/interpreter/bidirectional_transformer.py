import numpy as np
import re
import paddle

from .abc_interpreter import Interpreter, TransformerInterpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class BTCVInterpreter(TransformerInterpreter):
    """
    Bidirectional Transformer Interpreter.

    This is a specific interpreter for Transformers models, with two sub-processes: attentional perception, reasoning feedback.
    
    The following implementation is specially designed for Vision Transformer.
    """

    def __init__(self, paddle_model: callable, device: str = 'gpu:0', use_cuda=None) -> None:
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        TransformerInterpreter.__init__(self, paddle_model, device, use_cuda)

    def interpret(self,
                  inputs: str or list(str) or np.ndarray,
                  ap_mode: str = "head",
                  start_layer: int = 4,
                  steps: int = 20,
                  attn_map_name='^blocks.*.attn.attn_drop$', 
                  attn_v_name='^blocks.*.attn.qkv$',
                  attn_proj_name='^blocks.*.attn.proj$', 
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
        self._build_predict_fn(attn_map_name=attn_map_name, attn_v_name=attn_v_name, attn_proj_name=attn_proj_name)
        
        attns, grads, inputs, values, projs, preds = self.predict_fn(data)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        R = np.eye(s, s, dtype=attns[0].dtype)
        R = np.expand_dims(R, 0)
        
        if ap_mode == 'head':            
            for i, blk in enumerate(attns):
                if i < start_layer-1:
                    continue
                grad = grads[i]
                cam = blk
                cam = cam.reshape((-1, cam.shape[-1], cam.shape[-1]))
                grad = grad.reshape((-1, grad.shape[-1], grad.shape[-1]))

                Ih = np.mean(np.abs(np.matmul(np.transpose(cam, [0, 2, 1]), grad)), axis=(-1,-2))
                Ih = Ih.reshape([b, h])/np.sum(Ih, axis=-1)
                cam = np.matmul(Ih.reshape([b,1,h]),cam.reshape([b,h,-1])).reshape([b,s,s])

                R = R + np.matmul(cam, R)
        elif ap_mode == 'token':
            for i, blk in enumerate(attns):
                if i < start_layer-1:
                    continue
                cam = blk
                inp = inputs[i]
                v = np.transpose(values[i].reshape([b, s, 3, h, -1]), [2, 0, 1, 3, 4])[2]
                proj = projs[i]
                proj = np.expand_dims(proj, 0)
                proj = np.repeat(proj, repeats=b, axis=0)
                vproj = np.matmul(v.reshape([b, s, -1]), proj)
                
                order = np.linalg.norm(vproj, axis=-1).squeeze()/np.linalg.norm(inp, axis=-1).squeeze()
                m = np.diag(order)
                cam = cam.reshape([-1, cam.shape[-2], cam.shape[-1]]).mean(0)
            
                R = R + np.matmul(np.matmul(cam, m), R)
        else:
            assert "please specify the attentional perception mode"
                     
        
        total_gradients = np.zeros((b, h, s, s))
        for alpha in np.linspace(0, 1, steps):
            # forward propagation
            data_scaled = data * alpha
            _, gradients, _, _, _, _ = self.predict_fn(data_scaled, labels=label)

            total_gradients += gradients[-1]

        # gradient mean over heads.
        grad_head_mean = np.mean((total_gradients / steps).clip(min=0), axis=1)  # [b, s, s]

        if hasattr(self.paddle_model, 'global_pool') and self.paddle_model.global_pool:
            # For MAE ViT.
            explanation = (R * grad_head_mean)[:, 1:, :].mean(axis=1)
        else:
            explanation = R[:, 0, :] * grad_head_mean[:, 0, :]

        explanation = explanation[:, 1:].reshape((-1, 14, 14))

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

    def __init__(self, paddle_model: callable, device: str = 'gpu:0', use_cuda=None) -> None:
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        TransformerInterpreter.__init__(self, paddle_model, device, use_cuda)

    def interpret(self,
                  data: np.ndarray,
                  ap_mode: str = "head",
                  start_layer: int = 11,
                  steps: int = 20,
                  embedding_name='^ernie.embeddings.word_embeddings$', 
                  attn_map_name='^ernie.encoder.layers.*.self_attn.attn_drop$', 
                  attn_v_name='^ernie.encoder.layers.*.self_attn.v_proj$',
                  attn_proj_name='^ernie.encoder.layers.*.self_attn.out_proj$', 
                  label: int or None = None):
        """
        Args:
            data (str or list of strs or numpy.ndarray): The input text filepath or a list of filepaths or numpy
                array of read texts.
            ap_mode (str, default to head-wise): The approximation method of attentioanl perception stage,
                "head" for head-wise, "token" for token-wise. Default: ``head``.
            start_layer (int, optional): Compute the state from the start layer. Default: ``11``.
            steps (int, optional): number of steps in the Riemann approximation of the integral. Default: ``20``.
            embedding_name (str, optional): The layer name for embedding, head-wise/token-wise.
                Default: ``^ernie.embeddings.word_embeddings$``.
            attn_map_name (str, optional): The layer name to obtain the attention weights, head-wise/token-wise.
                Default: ``^ernie.encoder.layers.*.self_attn.attn_drop$``.
            attn_v_name (str, optional): The layer name for value projection, token-wise.
                Default: ``^ernie.encoder.layers.*.self_attn.v_proj$``.
            attn_proj_name (str, optional): The layer name for linear projection, token-wise.
                Default: ``ernie.encoder.layers.*.self_attn.out_proj$``.
            label (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels
                should be equal to the number of texts. If None, the most likely label for each text will be used.
                Default: ``None``.

        Returns:
            [numpy.ndarray]: interpretations for texts
        """

        b = data[0].shape[0]  # batch size
        assert b==1, "only support single sentence"
        self._build_predict_fn(embedding_name=embedding_name, attn_map_name=attn_map_name, 
                              attn_v_name=attn_v_name, attn_proj_name=attn_proj_name, nlp=True)
        
        attns, grads, inputs, values, projs, preds = self.predict_fn(data)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        R = np.eye(s, s, dtype=attns[0].dtype)
        R = np.expand_dims(R, 0)
        
        if ap_mode == 'head':            
            for i, blk in enumerate(attns):
                if i < start_layer-1:
                    continue
                grad = grads[i]
                cam = blk
                cam = cam.reshape((-1, cam.shape[-1], cam.shape[-1]))
                grad = grad.reshape((-1, grad.shape[-1], grad.shape[-1]))

                Ih = np.mean(np.abs(np.matmul(np.transpose(cam, [0, 2, 1]), grad)), axis=(-1,-2))
                Ih = Ih.reshape([b, h])/np.sum(Ih, axis=-1)
                cam = np.matmul(Ih.reshape([b,1,h]),cam.reshape([b,h,-1])).reshape([b,s,s])

                R = R + np.matmul(cam, R)
        elif ap_mode == 'token':
            for i, blk in enumerate(attns):
                if i < start_layer-1:
                    continue
                cam = blk
                inp = inputs[i]
                v = np.transpose(values[i].reshape([b, s, h, -1]), [0, 1, 2, 3])
                proj = projs[i]
                proj = np.expand_dims(proj, 0)
                proj = np.repeat(proj, repeats=b, axis=0)
                vproj = np.matmul(v.reshape([b, s, -1]), proj)
                
                order = np.linalg.norm(vproj, axis=-1).squeeze()/np.linalg.norm(inp, axis=-1).squeeze()
                m = np.diag(order)
                cam = cam.reshape([-1, cam.shape[-2], cam.shape[-1]]).mean(0)
            
                R = R + np.matmul(np.matmul(cam, m), R)
        else:
            assert "please specify the attentional perception mode"

        total_gradients = np.zeros((b, h, s, s))
        for alpha in np.linspace(0, 1, steps):
            # forward propagation
            _, gradients, _, _, _, _ = self.predict_fn(data, labels=label, alpha=alpha)
            total_gradients += gradients[-1]

        W_state = np.mean((total_gradients / steps).clip(min=0), axis=1)[:, 0, :].reshape((b, 1, s))

        explanation = (R * W_state)[:, 0, 1:]

        return explanation