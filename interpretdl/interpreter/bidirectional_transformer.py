import numpy as np
import re
import paddle

from .abc_interpreter import Interpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class BTInterpreter(Interpreter):
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
        Interpreter.__init__(self, paddle_model, device, use_cuda)

    def interpret(self,
                  inputs: str or list(str) or np.ndarray,
                  ap_mode: str = "head",
                  start_layer: int = 4,
                  steps: int = 20,
                  label: int or None = None,
                  resize_to: int = 224,
                  crop_to: int or None = None,
                  visual: bool = True,
                  save_path: str or None = None):
        """
        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy 
                array of read images.
            ap_mode (str, default to head-wise): The approximation method of attentioanl perception stage, "head" for head-wise, "token" for token-wise.
            start_layer (int, optional): Compute the state from the start layer. Default: ``4``.
            steps (int, optional): number of steps in the Riemann approximation of the integral. Default: ``20``.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels 
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
        self._build_predict_fn()
        
        attns, grads, inputs, values, projs, preds = self.predict_fn(data)
        assert start_layer < len(attns), "start_layer should be in the range of [0, num_block-1]"

        if label is None:
            label = preds

        b, h, s, _ = attns[0].shape
        num_blocks = len(attns)
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
                grad = grads[i]
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
            _, gradients, _, _, _, _ = self.predict_fn(data_scaled, label=label)

            total_gradients += gradients[-1]

        W_state = np.mean((total_gradients / steps).clip(min=0), axis=1)[:, 0, :].reshape((b, 1, s))

        explanation = (R * W_state)[:, 0, 1:].reshape((-1, 14, 14))

        # visualization and save image.
        vis_explanation = explanation_to_vis(imgs, explanation[0], style='overlay_heatmap')
        if visual:
            show_vis_explanation(vis_explanation)
        if save_path is not None:
            save_image(save_path, vis_explanation)

        return explanation

    def _build_predict_fn(self, rebuild: bool = False):
        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        if self.predict_fn is None or rebuild:
            import paddle
            self._paddle_env_setup()  # inherit from InputGradientInterpreter

            def predict_fn(data, label=None):
                data = paddle.to_tensor(data)
                data.stop_gradient = False

                attns = []
                def attn_hook(layer, input, output):
                    attns.append(output)
                    
                inputs = []
                def input_hook(layer, input):
                    inputs.append(input)
                    
                values = []
                projs = []
                def v_hook(layer, input, output):
                    values.append(output)
                    

                hooks = []
                for n, v in self.paddle_model.named_sublayers():
                    if re.match('^blocks.*.attn.attn_drop$', n):
                        h = v.register_forward_post_hook(attn_hook)
                        hooks.append(h)
                    elif re.match('^blocks.*.attn.proj$', n):
                        projs.append(v.weight)
                    elif re.match('^blocks.*.attn.qkv$', n):
                        h = v.register_forward_pre_hook(input_hook)
                        hooks.append(h)
                        h = v.register_forward_post_hook(v_hook)
                        hooks.append(h)
                out = self.paddle_model(data)
                for h in hooks:
                    h.remove()

                out = paddle.nn.functional.softmax(out, axis=1)
                preds = paddle.argmax(out, axis=1)
                if label is None:
                    label = preds.numpy()

                attns_grads = []
                
                label_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(label), num_classes=out.shape[1])
                target = paddle.sum(out * label_onehot, axis=1)
                target.backward()
                for i, blk in enumerate(attns):
                    grad = blk.grad.numpy()
                    attns_grads.append(grad)
                    attns[i] = blk.numpy()
                target.clear_gradient()
                
                for i, blk in enumerate(inputs):
                    inputs[i] = blk[0].numpy()
                
                for i, blk in enumerate(values):
                    values[i] = blk[0].numpy()
                
                for i, blk in enumerate(projs):
                    projs[i] = blk.numpy()
                
                return attns, attns_grads, inputs, values, projs, label

            self.predict_fn = predict_fn