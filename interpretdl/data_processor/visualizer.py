from skimage.segmentation import quickshift, mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import cv2


def show_vis_explanation(explanation_image, cmap=None):
    """
    Get the environment and show the image.
    Args:
        explanation_image:
        cmap:

    Returns: None

    """
    plt.imshow(explanation_image, cmap)
    plt.axis("off")
    plt.show()


# def is_jupyter():
#     # ref: https://stackoverflow.com/a/39662359/4834515
#     try:
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell':
#             return True  # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False  # Probably standard Python interpreter


def explanation_to_vis(batched_image: np.ndarray, explanation: np.ndarray, style='grayscale') -> np.ndarray:
    """

    Args:
        batched_image: e.g., (1, height, width, 3).
        explanation: should have the same width and height as image.
        style: ['grayscale', 'heatmap', 'overlay_grayscale', 'overlay_heatmap', 'overlay_threshold'].

    Returns:

    """
    if len(batched_image.shape) == 4:
        assert batched_image.shape[0] == 1, "For one image only"
        batched_image = batched_image[0]
        assert len(batched_image.shape) == 3

    assert len(explanation.shape) == 2, f"image shape {batched_image.shape} vs " \
                                        f"explanation {explanation.shape}"

    image = batched_image
    if style == 'grayscale':
        # explanation has the same size as image, no need to scale.
        # usually for gradient-based explanations w.r.t. the image.
        return _grayscale(explanation)
    elif style == 'heatmap':
        # explanation's width and height are usually smaller than image.
        # usually for CAM, GradCAM etc, which produce lower-resolution explanations.
        return _heatmap(explanation, (image.shape[1], image.shape[0]))  # image just for the shape.
    elif style == 'overlay_grayscale':
        return overlay_grayscale(image, explanation)
    elif style == 'overlay_heatmap':
        return overlay_heatmap(image, explanation)
    elif style == 'overlay_threshold':
        # usually for LIME etc, which originally shows positive and negative parts.
        return overlay_threshold(image, explanation)
    else:
        raise KeyError("Unknown visualization style.")


def _grayscale(explanation: np.ndarray, percentile=99) -> np.ndarray:
    """

    Args:
        explanation: numpy.ndarray, 2d.
        percentile:

    Returns: numpy.ndarray, uint8, same shape as explanation

    """
    assert len(explanation.shape) == 2, f"{explanation.shape}. " \
                                        "Currently support 2D explanation results for visualization. " \
                                        "Reduce higher dimensions to 2D for visualization."

    assert isinstance(percentile, int)
    assert 0 <= percentile <= 100                       

    image_2d = explanation

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    x = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1) * 255
    x = np.uint8(x)

    return x


def overlay_grayscale(image, explanation, percentile=99) -> np.ndarray:
    x = _grayscale(explanation, percentile)

    overlay_vis = np.zeros_like(image, dtype=np.uint8)
    overlay_vis[:, :, 1] = x

    overlay_vis = overlay_vis * 0.4 + image * 0.6

    return np.uint8(overlay_vis)


def _heatmap(explanation, resize_shape=(224, 224)) -> np.ndarray:
    """

    Args:
        explanation:
        resize_shape: (width, height)

    Returns:

    """
    assert len(explanation.shape) == 2, f"{explanation.shape}. " \
                                        f"Currently support 2D explanation results for visualization. " \
                                        "Reduce higher dimensions to 2D for visualization."

    # explanation = np.maximum(explanation, 0)
    # ex_max = np.max(explanation)
    # explanation /= ex_max
    
    explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min()) 

    explanation = cv2.resize(explanation, resize_shape)
    explanation = np.uint8(255 * explanation)
    explanation = cv2.applyColorMap(explanation, cv2.COLORMAP_JET)
    explanation = cv2.cvtColor(explanation, cv2.COLOR_BGR2RGB)

    return explanation


def overlay_heatmap(image, explanation) -> np.ndarray:
    x = _heatmap(explanation, (image.shape[1], image.shape[0]))

    overlay_vis = x * 0.4 + image * 0.6

    return np.uint8(overlay_vis)


def overlay_threshold(image, explanation_mask) -> np.ndarray:
    overlay_vis = np.zeros_like(image, dtype=np.uint8)
    overlay_vis[:, :, 1] = explanation_mask * 255

    overlay_vis = overlay_vis * 0.6 + image * 0.4

    return np.uint8(overlay_vis)


def sp_to_array(segments, sp_weights_list):
    explanation_mask = np.zeros(segments.shape, np.float32)
    for sp_i, sp_w in sp_weights_list:
        explanation_mask[segments == sp_i] = sp_w
    return explanation_mask


def sp_weights_to_image_explanation(image, sp_weights, label=None, segments=None, ratio_superpixels=0.2):
    """Convert lime superpixel weights to an array, with thresholding.

    Args:
        image (numpy.ndarray): _description_
        sp_weights (dict): a dict of tuples {class_index: [(sp_index, sp_weight)]}.
        label (int, optional): _description_. Defaults to None.
        segments (numpy.ndarray, optional): _description_. Defaults to None.
        ratio_superpixels (float, optional): _description_. Defaults to 0.2.

    Raises:
        KeyError: _description_

    Returns:
        numpy.ndarray: _description_
    """
    if label is None:
        label = list(sp_weights.keys())[0]

    if label not in sp_weights:
        raise KeyError('Label not in interpretation')

    if segments is None:
        segments = quickshift(image, sigma=1)

    num_sp = int(ratio_superpixels * len(sp_weights[label]))
    sp_weight = sp_weights[label]
    explanation_mask = np.zeros(segments.shape, segments.dtype)

    fs = [x[0] for x in sp_weight if x[1] > 0][:num_sp]
    for f in fs:
        explanation_mask[segments == f] = 1

    return explanation_mask


def save_image(file_path, image):
    plt.imsave(file_path, image)


class VisualizationTextRecord:
    """
    A record for text visulization.
    Part of the code is modified from https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
    """

    def __init__(
            self,
            words,
            word_importances,
            true_label,
            pred_class,
            pred_prob,
            interp_class, ):
        self.word_importances = word_importances
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_label = true_label
        self.interp_class = interp_class
        self.words = words

    def record_html(self):
        return "".join([
            "<tr>",
            self._format_class(self.true_label),
            self._format_class(self.pred_class, self.pred_prob),
            self._format_class(self.interp_class),
            self._format_word_importances(),
            "<tr>",
        ])

    def _format_class(self, label, prob=None):
        if prob is None:
            return '<td><text style="padding-right:2em"><b>{label}</b></text></td>'.format(
                label=label)
        else:
            return '<td><text style="padding-right:2em"><b>{label} ({prob:.2f})</b></text></td>'\
        .format(label=label, prob=prob)

    def _format_word_importances(self):
        tags = ["<td>"]
        for word, importance in zip(self.words,
                                    self.word_importances[:len(self.words)]):
            color = self._background_color(importance)
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>'.format(
                color=color, word=word)
            tags.append(unwrapped_tag)
        tags.append("</td>")
        return "".join(tags)

    def _background_color(self, importance):
        importance = max(-1, min(1, importance))
        if importance > 0:
            hue = 120
            sat = 75
            lig = 100 - int(50 * importance)
        else:
            hue = 0
            sat = 75
            lig = 100 - int(-40 * importance)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def visualize_text(text_records):
    html = ["<table width: 100%>"]
    rows = [
        "<tr><th>True Label</th>"
        "<th>Predicted Label (Prob)</th>"
        "<th>Target Label</th>"
        "<th>Word Importance</th>"
    ]
    for record in text_records:
        rows.append(record.record_html())
    html.append("".join(rows))
    html.append("</table>")
    display(HTML("".join(html)))
