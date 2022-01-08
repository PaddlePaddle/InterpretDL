import os
import sys
import cv2
import numpy as np
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


# def read_image_to_np(img_path: str) -> np.ndarray:
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB.
#     # h, w, c = img.shape
#     # assert c in [1, 3]
#     # assert c.dytpe == np.unit8
#     return img


def resize_image(img: np.ndarray, target_size: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """resize image with shorter edge equal to target_size.

    Args:
        img: image data
        target_size: resize short target size
        interpolation: interpolation mode

    Returns:
        resized image data
    """

    h, w, c = img.shape
    assert c in [1, 3]

    percent = float(target_size) / min(h, w)
    resized_width = int(round(w * percent))
    resized_height = int(round(h * percent))

    resized = cv2.resize(img, (resized_width, resized_height), interpolation=interpolation)
    # assert resized.dytpe == np.float32
    return resized


def crop_image(img: np.ndarray, target_size: int, center=True) -> np.ndarray:
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


def preprocess_image(img: np.ndarray, random_mirror=False) -> np.ndarray:
    """
    image(uint8) to tensor(float32). scaled by 1/255, centered, standarized.
    :param img: np.ndarray: shape: [ns, h, w, 3], color order: rgb.
    :return: np.ndarray: shape: [ns, 3, h, w]
    """
    # ImageNet stats.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transpose from [ns, h, w, 3] to [ns, c, h, w], and scaled by 1/255
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


def read_image(img_path, target_size=256, crop_size=224, crop=True) -> np.ndarray:
    """
    resize_short to target_size, then center crop to crop_size or not crop.
    :param img_path: one image path
    :return: np.ndarray: shape: [1, h, w, 3], dtype: uint8, color order: rgb.
    """

    if isinstance(img_path, str):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.array(img)
            img = resize_image(img, target_size)
            if crop:
                img = crop_image(img, target_size=crop_size, center=True)
            img = np.expand_dims(img, axis=0)
            return img
    elif isinstance(img_path, np.ndarray):
        assert len(img_path.shape) == 4
        return img_path
    else:
        ValueError(f"Not recognized data type {type(img_path)}.")


def restore_image(float_input_data: np.ndarray) -> np.ndarray:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    float_input_data *= img_std
    float_input_data += img_mean
    float_input_data *= 255
    float_input_data += 0.5  # for float to integer
    img = np.uint8(float_input_data.transpose((0, 2, 3, 1)))
    return img


# def _find_classes(dir):
#     # Faster and available in Python 3.5 and above
#     classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx


# def get_typical_dataset_info(dataset_dir,
#                              subset="test",
#                              shuffle=False,
#                              random_seed=None):
#     """
#     where {dataset_dir}/{train,test,segmentations}}/{class1, class2, ...}/*.png exists.

#     segmentations are optional.

#     Args:
#         dataset_dir:
#         shuffle:
#         random_seed:

#     Returns:

#     """
#     IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
#                       '.tiff', '.webp')

#     # read
#     set_dir = os.path.join(dataset_dir, subset)
#     seg_dir = os.path.join(dataset_dir, 'segmentations')

#     class_names, class_to_idx = _find_classes(set_dir)
#     # num_classes = len(class_names)
#     image_paths = []
#     seg_paths = []
#     labels = []
#     for class_name in sorted(class_names):
#         classes_dir = os.path.join(set_dir, class_name)
#         for img_path in sorted(glob.glob(os.path.join(classes_dir, '*'))):
#             if not img_path.lower().endswith(IMG_EXTENSIONS):
#                 continue

#             image_paths.append(img_path)
#             seg_paths.append(
#                 os.path.join(seg_dir,
#                              img_path.split('test/')[-1].replace('jpg',
#                                                                  'png')))
#             labels.append(class_to_idx[class_name])

#             assert os.path.exists(seg_paths[-1]), seg_paths[-1]

#     image_paths = np.array(image_paths)
#     seg_paths = np.array(seg_paths)
#     labels = np.array(labels)

#     if shuffle:
#         np.random.seed(random_seed)
#         random_per = np.random.permutation(range(len(image_paths)))
#         image_paths = image_paths[random_per]
#         seg_paths = seg_paths[random_per]
#         labels = labels[random_per]

#     return image_paths, seg_paths, labels, len(class_names)


# def extract_img_paths(directory):
#     IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
#                       '.tiff', '.webp')
#     img_paths = []
#     img_names = []
#     for file in os.listdir(directory):
#         if file.lower().endswith(IMG_EXTENSIONS):
#             img_paths.append(os.path.join(directory, file))
#             img_names.append(file)
#     return img_paths, img_names



def images_transform_pipeline(array_or_path, resize_to=224, crop_to=None):
    """[summary]

    Args:
        array_or_path ([type]): [description]
        resize_to (int, optional): [description]. Defaults to 224.
        crop_to ([type], optional): [description]. Defaults to None.
    """
    if crop_to is not None:
        assert isinstance(crop_to, int)
        def read_image_func(path):
            return read_image(path, target_size=resize_to, crop_size=crop_to)
    else:
        def read_image_func(path):
            return read_image(path, target_size=resize_to, crop=False)

    if isinstance(array_or_path, str):
        # one single image path.
        uint8_imgs = read_image_func(array_or_path)
        float_input_data = preprocess_image(uint8_imgs)
    elif isinstance(array_or_path, list) and all(
            isinstance(elem, str) for elem in array_or_path):
        # a list of image paths.
        uint8_imgs = []
        for fp in array_or_path:
            uint8_img = read_image_func(fp)
            uint8_imgs.append(uint8_img)
        uint8_imgs = np.concatenate(uint8_imgs)
        float_input_data = preprocess_image(uint8_imgs)
    else:
        # an array. will not resize nor crop.
        if len(array_or_path.shape) == 3:
            uint8_imgs = np.expand_dims(array_or_path, axis=0)
        elif len(array_or_path.shape) == 4:
            uint8_imgs = array_or_path

        if np.issubdtype(array_or_path.dtype, np.integer):
            # array_or_path is an image.
            uint8_imgs = uint8_imgs.copy()
            float_input_data = preprocess_image(uint8_imgs)
        else:
            # array_or_path is float input data.
            uint8_imgs = restore_image(array_or_path.copy())
            float_input_data = array_or_path

    return uint8_imgs, float_input_data


# def preprocess_inputs(inputs, model_input_shape):
#     """[summary]

#     Args:
#         inputs ([type]): can be a str, a list of str, or np.ndarray.
#         model_input_shape ([type]): [description]

#     Returns:
#         imgs: uint8 images, used for visualization. [ns, h, w, 3]
#         data: float scaled image data, used for computation. [ns, 3, h, w]
#     """
#     if isinstance(inputs, str):
#         imgs = read_image(inputs, crop_size=model_input_shape[1])
#         data = preprocess_image(imgs)
#     elif isinstance(inputs, list) and all(
#             isinstance(elem, str) for elem in inputs):
#         imgs = []
#         for fp in inputs:
#             img = read_image(fp, crop_size=model_input_shape[1])
#             imgs.append(img)
#         imgs = np.concatenate(imgs)
#         data = preprocess_image(imgs)
#     else:
#         if len(inputs.shape) == 3:
#             inputs = np.expand_dims(inputs, axis=0)
#         if np.issubdtype(inputs.dtype, np.integer):
#             imgs = inputs.copy()
#             data = preprocess_image(inputs)
#         else:
#             imgs = restore_image(inputs.copy())
#             data = inputs
#     return imgs, data


def preprocess_save_path(save_path, bsz):
    if isinstance(save_path, str):
        save_path = [save_path]
    if save_path is None:
        save_path = [None] * bsz
    assert len(
        save_path
    ) == bsz, f"number of save_paths ({len(save_path)}) should be equal to number of images ({bsz})"
    return save_path
