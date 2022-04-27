import numpy as np

from .abc_interpreter import IntermediateLayerInterpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class RolloutInterpreter(IntermediateLayerInterpreter):
    """
    Rollout Interpreter.
    
    This is a specific interpreter for Transformers models.
    RolloutInterpreter assumes that attentions can be linearly combined and the obtained score is able to show the 
    scores of tokens, which gives an explanation of token importance.

    More details regarding the Rollout method can be found in the original paper:
    https://arxiv.org/abs/2005.00928.
    """

    def __init__(self, paddle_model: callable, device: str = 'gpu:0', use_cuda=None) -> None:
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        IntermediateLayerInterpreter.__init__(self, paddle_model, device, use_cuda)

    def interpret(self,
                  inputs: str or list(str) or np.ndarray,
                  start_layer: int = 0,
                  attention_layer_pattern: str = '^blocks.*.attn.attn_drop$',
                  resize_to: int = 224,
                  crop_to: int or None = None,
                  visual: bool = True,
                  save_path: str or None = None):
        """        
        Given ``inputs``, RolloutInterpreter obtains all attention maps (of layers whose name matches 
        ``attention_layer_pattern``) and calculates their matrix multiplication. The ``start_layer`` controls the
        number of involved layers.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy 
                array of read images.
            start_layer (int, optional): The index of the start layer involving the computation of attentions. Defaults
                to ``0``.
            attention_layer_pattern (str, optional): the string pattern to pick the layers that match the pattern. 
                Defaults to ``^blocks.*.attn.attn_drop$``.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The filepath(s) to save the processed image(s). If None, the image will not be 
                saved. Default: ``None``.

        Returns:
            [np.ndarray]: interpretations/heatmap for images.
        """

        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)
        bsz = len(data)  # batch size
        save_path = preprocess_save_path(save_path, bsz)

        self._build_predict_fn(target_layer_pattern=attention_layer_pattern)
        blk_attns, _, _ = self.predict_fn(data)

        assert start_layer < len(blk_attns), "start_layer should be in the range of [0, num_block-1]"

        all_layer_attentions = []
        for attn_heads in blk_attns:
            avg_heads = attn_heads.sum(axis=1) / attn_heads.shape[1]
            all_layer_attentions.append(avg_heads)

        # compute rollout between attention layers
        # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
        num_tokens = all_layer_attentions[0].shape[1]

        eye = np.eye(num_tokens)
        all_layer_attentions = [all_layer_attentions[i] + eye for i in range(len(all_layer_attentions))]
        matrices_aug = [
            all_layer_attentions[i] / all_layer_attentions[i].sum(axis=-1, keepdims=True)
            for i in range(len(all_layer_attentions))
        ]
        joint_attention = matrices_aug[start_layer]
        for i in range(start_layer + 1, len(matrices_aug)):
            joint_attention = matrices_aug[i] @ joint_attention

        # hard coding: 14, 14
        rollout_explanation = joint_attention[:, 0, 1:].reshape((-1, 14, 14))

        # visualization and save image.
        for i in range(bsz):
            vis_explanation = explanation_to_vis(imgs[i], rollout_explanation[i], style='overlay_heatmap')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return rollout_explanation
