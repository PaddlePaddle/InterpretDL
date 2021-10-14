
import cv2
import numpy as np
import paddle
from tqdm import tqdm

from .abc_interpreter import IntermediateLayerInterpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class ScoreCAMInterpreter(IntermediateLayerInterpreter):
    """
    Score CAM Interpreter.

    More details regarding the Score CAM method can be found in the original paper:
    https://arxiv.org/abs/1910.01279
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=None,
                 device='gpu:0',
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradCAMInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        IntermediateLayerInterpreter.__init__(self, paddle_model, device, use_cuda)
        self.model_input_shape = model_input_shape

    def interpret(self,
                  inputs,
                  target_layer_name,
                  labels=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            target_layer_name (str): The target layer to calculate gradients.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/heatmap for each image
        :rtype: numpy.ndarray
        """

        imgs, data = preprocess_inputs(inputs, self.model_input_shape)

        bsz = len(data)
        save_path = preprocess_save_path(save_path, bsz)

        b, c, h, w = data.shape

        self._build_predict_fn(target_layer=target_layer_name)

        if labels is None:
            _, probs, labels = self.predict_fn(data)

        labels = np.array(labels).reshape((bsz, ))
        feature_map, _, _ = self.predict_fn(data)
        interpretations = np.zeros((b, h, w))

        for i in tqdm(range(feature_map.shape[1]), leave=False, position=1):
            feature_channel = feature_map[:, i, :, :]
            feature_channel = np.concatenate([
                np.expand_dims(cv2.resize(f, (h, w)), 0)
                for f in feature_channel
            ])
            norm_feature_channel = np.array(
                [(f - f.min()) / (f.max() - f.min()) if f.max() - f.min() > 0.0 else f
                 for f in feature_channel]).reshape((b, 1, h, w))
            _, probs, _ = self.predict_fn(data * norm_feature_channel)
            scores = [p[labels[i]] for i, p in enumerate(probs)]
            interpretations += feature_channel * np.array(scores).reshape((
                b, ) + (1, ) * (interpretations.ndim - 1))

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
        for i in range(b):
            vis_explanation = explanation_to_vis(imgs[i], interpretations[i], style='overlay_heatmap')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return interpretations
