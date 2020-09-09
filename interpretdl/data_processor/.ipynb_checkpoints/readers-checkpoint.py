import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import six
import glob
from PIL import Image
import pickle


def load_npy_dict_file(fname):
    if fname is None:
        return None

    if os.path.exists(fname):
        npy_dict = np.load(fname, allow_pickle=True).item()
        assert type(npy_dict) == dict or npy_dict is None
        return npy_dict
    else:
        return None


def load_pickle_file(fname):
    if fname is None:
        return None

    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            model = pickle.load(f)

        return model
    else:
        return None


def resize_short(img, target_size, interpolation=None):
    """resize image

    Args:
        img: image data
        target_size: resize short target size
        interpolation: interpolation mode

    Returns:
        resized image data
    """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    if interpolation:
        resized = cv2.resize(
            img, (resized_width, resized_height), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center=True):
    """crop image

    Args:
        img: images data
        target_size: crop target size
        center: crop mode

    Returns:
        img: cropped image data
    """
    height, width = img.shape[:2]
    size = target_size
    if center:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def preprocess_image(img, random_mirror=False):
    """
    centered, scaled by 1/255.
    :param img: np.array: shape: [ns, h, w, 3], color order: rgb.
    :return: np.array: shape: [ns, h, w, 3]
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transpose to [ns, 3, h, w]
    img = img.astype('float32').transpose((0, 3, 1, 2)) / 255

    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    if random_mirror:
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img[:, :, ::-1, :]

    return img


def read_image(img_path, target_size=256, crop_size=224):
    """
    resize_short to 256, then center crop to 224.
    :param img_path: one image path
    :return: np.array: shape: [1, h, w, 3], color order: rgb.
    """

    if isinstance(img_path, str):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.array(img)
            img = resize_short(img, target_size, interpolation=None)
            img = crop_image(img, target_size=crop_size, center=True)
            # img = img[:, :, ::-1]
            img = np.expand_dims(img, axis=0)
            return img
    elif isinstance(img_path, np.ndarray):
        assert len(img_path.shape) == 4
        return img_path
    else:
        ValueError(f"Not recognized data type {type(img_path)}.")


def _find_classes(dir):
    # Faster and available in Python 3.5 and above
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_typical_dataset_info(dataset_dir,
                             subset="test",
                             shuffle=False,
                             random_seed=None):
    """
    where {dataset_dir}/{train,test,segmentations}}/{class1, class2, ...}/*.png exists.

    segmentations are optional.

    Args:
        dataset_dir:
        shuffle:
        random_seed:

    Returns:

    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                      '.tiff', '.webp')

    # read
    set_dir = os.path.join(dataset_dir, subset)
    seg_dir = os.path.join(dataset_dir, 'segmentations')

    class_names, class_to_idx = _find_classes(set_dir)
    # num_classes = len(class_names)
    image_paths = []
    seg_paths = []
    labels = []
    for class_name in sorted(class_names):
        classes_dir = os.path.join(set_dir, class_name)
        for img_path in sorted(glob.glob(os.path.join(classes_dir, '*'))):
            if not img_path.lower().endswith(IMG_EXTENSIONS):
                continue

            image_paths.append(img_path)
            seg_paths.append(
                os.path.join(seg_dir,
                             img_path.split('test/')[-1].replace('jpg',
                                                                 'png')))
            labels.append(class_to_idx[class_name])

            assert os.path.exists(seg_paths[-1]), seg_paths[-1]

    image_paths = np.array(image_paths)
    seg_paths = np.array(seg_paths)
    labels = np.array(labels)

    if shuffle:
        np.random.seed(random_seed)
        random_per = np.random.permutation(range(len(image_paths)))
        image_paths = image_paths[random_per]
        seg_paths = seg_paths[random_per]
        labels = labels[random_per]

    return image_paths, seg_paths, labels, len(class_names)


def restore_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img *= img_std
    img += img_mean
    img *= 255
    img = np.uint8(img.transpose((0, 2, 3, 1)))
    return img


def extract_img_paths(directory):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                      '.tiff', '.webp')
    img_paths = []
    img_names = []
    for file in os.listdir(directory):
        if file.lower().endswith(IMG_EXTENSIONS):
            img_paths.append(os.path.join(directory, file))
            img_names.append(file)
    return img_paths, img_names


def preprocess_inputs(inputs, save_path, model_input_shape):
    if isinstance(inputs, str):
        imgs = read_image(inputs, crop_size=model_input_shape[1])
        data = preprocess_image(imgs)
        if save_path is None or isinstance(save_path, str):
            save_path = [save_path]
    elif bool(list) and isinstance(inputs, list) and all(
            isinstance(elem, str) for elem in inputs):
        imgs = []
        for fp in inputs:
            img = read_image(fp, crop_size=model_input_shape[1])
            imgs.append(img)
        imgs = np.concatenate(imgs)
        data = preprocess_image(imgs)
        if save_path is None:
            save_path = [None] * len(imgs)
    else:
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, axis=0)
        if np.issubdtype(inputs.dtype, np.integer):
            imgs = inputs.copy()
            data = preprocess_image(inputs)
        else:
            imgs = restore_image(inputs.copy())
            data = inputs
        if save_path is None:
            save_path = [None] * len(imgs)
    return imgs, data, save_path
