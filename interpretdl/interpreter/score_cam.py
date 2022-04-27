import cv2
import numpy as np
from tqdm import tqdm

from .abc_interpreter import IntermediateLayerInterpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class ScoreCAMInterpreter(IntermediateLayerInterpreter):
    """
    Score-CAM Interpreter.

    ScoreCAMInterpreter bridges the gap between perturbation-based and CAM-based methods, and derives the weight of 
    activation maps in an intuitively understandable way.

    More details regarding the Score CAM method can be found in the original paper:
    https://arxiv.org/abs/1910.01279.
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
                  target_layer_name: str,
                  labels: list or np.ndarray = None,
                  resize_to: int = 224,
                  crop_to: int or None = None,
                  visual: bool = True,
                  save_path: str = None) -> np.ndarray:
        """
        Main function of the interpreter.

        (TODO) The technical details will be described later.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy 
                array of read images.
            target_layer_name (str): The target layer to calculate gradients.
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
            [np.ndarray]: interpretations/heatmap for images
        """

        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)

        bsz, c, h, w = data.shape
        save_path = preprocess_save_path(save_path, bsz)

        self._build_predict_fn(target_layer=target_layer_name)

        if labels is None:
            _, probs, labels = self.predict_fn(data)

        labels = np.array(labels).reshape((bsz, ))
        feature_maps, _, _ = self.predict_fn(data)
        feature_map = feature_maps[0]
        interpretations = np.zeros((bsz, h, w))

        for i in tqdm(range(feature_map.shape[1]), leave=True, position=0):
            feature_channel = feature_map[:, i, :, :]
            feature_channel = np.concatenate([np.expand_dims(cv2.resize(f, (w, h)), 0) for f in feature_channel])
            norm_feature_channel = np.array([(f - f.min()) / (f.max() - f.min()) if f.max() - f.min() > 0.0 else f
                                             for f in feature_channel]).reshape((bsz, 1, h, w))
            _, probs, _ = self.predict_fn(data * norm_feature_channel)
            scores = [p[labels[i]] for i, p in enumerate(probs)]
            interpretations += feature_channel * np.array(scores).reshape((bsz, ) + (1, ) * (interpretations.ndim - 1))

        # interpretations = np.maximum(interpretations, 0)
        # interpretations_min, interpretations_max = interpretations.min(
        # ), interpretations.max()

        # if interpretations_min == interpretations_max:
        #     return None

        # interpretations = (interpretations - interpretations_min) / (
        #     interpretations_max - interpretations_min)

        # interpretations = np.array([(interp - interp.min()) /
        #                             (interp.max() - interp.min())
        #                             for interp in interpretations])

        # visualization and save image.
        for i in range(bsz):
            vis_explanation = explanation_to_vis(imgs[i], interpretations[i], style='overlay_heatmap')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return interpretations
