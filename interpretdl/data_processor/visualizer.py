from skimage.segmentation import quickshift, mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
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
    gradients = gradients[0].transpose((1, 2, 0))
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
    interpretation = np.clip(0.7 * img[0] + 0.5 * interpretation, 0, 255)

    x = np.uint8(interpretation)
    x = Image.fromarray(x)

    if visual:
        visualize_image(x)

    if save_path is not None:
        x.save(save_path)


def visualize_grayscale(gradients, percentile=99, visual=True, save_path=None):
    image_2d = np.sum(np.abs(gradients[0]), axis=0)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    x = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1) * 255
    x = np.uint8(x)
    x = Image.fromarray(x)
    if visual:
        visualize_image(x)

    if save_path is not None:
        x.save(save_path)


def visualize_gradcam(feature_map, gradients, org, visual=True,
                      save_path=None):
    # take the average of gradient for each channel
    mean_g = np.mean(gradients, (1, 2))
    heatmap = feature_map.transpose([1, 2, 0])
    # multiply the feature map by average gradients
    for i in range(len(mean_g)):
        heatmap[:, :, i] *= mean_g[i]

    heatmap = np.mean(heatmap, axis=-1)
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    org = np.array(org).astype('float32')
    org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (org.shape[1], org.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    x = heatmap * 0.8 + org
    if visual:
        display.display(display.Image(x))

    if save_path is not None:
        cv2.imwrite(save_path, x)
