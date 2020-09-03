from skimage.segmentation import quickshift, mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from PIL import Image
import cv2


def is_jupyter():
    # ref: https://stackoverflow.com/a/39662359/4834515
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def show_important_parts(image,
                         lime_weights,
                         label=None,
                         segments=None,
                         ratio_superpixels=0.2):
    if label is None:
        label = list(lime_weights.keys())[0]

    if label not in lime_weights:
        raise KeyError('Label not in interpretation')

    if segments is None:
        segments = quickshift(image, sigma=1)

    num_sp = int(ratio_superpixels * len(lime_weights[label]))
    lime_weight = lime_weights[label]
    mask = np.zeros(segments.shape, segments.dtype)
    temp = image.copy()

    fs = [x[0] for x in lime_weight if x[1] > 0][:num_sp]
    for f in fs:
        temp[segments == f, 1] = 255
        mask[segments == f] = 1

    return mark_boundaries(temp, mask)


def visualize_image(image):
    if is_jupyter():
        import IPython.display as display
        display.display(display.Image(image))
    else:
        plt.imshow(image)
        plt.show()


def save_image(file_path, image):
    plt.imsave(file_path, image)


def visualize_overlay(gradients, img, visual=True, save_path=None):
    gradients = gradients.transpose((1, 2, 0))
    interpretation = np.clip(gradients, 0, 1)
    channel = [0, 255, 0]
    interpretation = np.average(interpretation, axis=2)

    m, e = np.percentile(np.abs(interpretation),
                         99.5), np.min(np.abs(interpretation))
    transformed = (np.abs(interpretation) - e) / (m - e)

    # Recover the original sign of the interpretation.
    transformed *= np.sign(interpretation)

    # Clip values above and below.
    transformed = np.clip(transformed, 0.0, 1.0)

    interpretation = np.expand_dims(transformed, 2) * channel
    interpretation = np.clip(0.7 * img + 0.5 * interpretation, 0, 255)

    x = np.uint8(interpretation)
    x = Image.fromarray(x)

    if visual:
        visualize_image(x)

    if save_path is not None:
        x.save(save_path)


def visualize_grayscale(gradients, percentile=99, visual=True, save_path=None):
    image_2d = np.sum(np.abs(gradients), axis=0)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    x = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1) * 255
    x = np.uint8(x)
    x = Image.fromarray(x)
    if visual:
        visualize_image(x)

    if save_path is not None:
        x.save(save_path)


def visualize_heatmap(heatmap, org, visual=True, save_path=None):
    org = np.array(org).astype('float32')
    org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (org.shape[1], org.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    x = heatmap * 0.5 + org * 0.7
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    #x = Image.fromarray(x)

    if visual:
        visualize_image(x)

    if save_path is not None:
        #x.save(save_path)
        cv2.imwrite(save_path, x)


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
