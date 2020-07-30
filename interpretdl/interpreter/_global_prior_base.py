import os
import os.path as osp
import numpy as np
import tqdm
from sklearn.linear_model import Ridge

from ..data_processor.readers import read_image, load_pickle_file
from ..common.paddle_utils import FeatureExtractor, extract_superpixel_features, get_pre_models


def data_labels(file_path_list, predict_fn, batch_size):
    print(
        "Initialization for fast NormLIME: Computing each sample in the test list."
    )

    _, h_pre_models_kmeans = get_pre_models()
    kmeans_model = load_pickle_file(h_pre_models_kmeans)
    num_features = len(kmeans_model.cluster_centers_)
    y_labels = []

    fextractor = FeatureExtractor()
    x_data = []
    tmp_imgs = []

    for each_data_ in tqdm.tqdm(file_path_list):
        image_show = read_image(each_data_)
        tmp_imgs.append(image_show)

        feature = fextractor.forward(image_show).transpose((1, 2, 0))
        # print(time.time() - end)  # 40 % time.
        segments = np.zeros((image_show.shape[1], image_show.shape[2]),
                            np.int32)
        num_blocks = 10
        height_per_i = segments.shape[0] // num_blocks + 1
        width_per_i = segments.shape[1] // num_blocks + 1

        for i in range(segments.shape[0]):
            for j in range(segments.shape[1]):
                segments[i,
                         j] = i // height_per_i * num_blocks + j // width_per_i

        X = extract_superpixel_features(feature, segments)

        try:
            cluster_labels = kmeans_model.predict(X)
        except AttributeError:
            from sklearn.metrics import pairwise_distances_argmin_min
            cluster_labels, _ = pairwise_distances_argmin_min(
                X, kmeans_model.cluster_centers_)
        x_data_i = np.zeros((num_features))
        for c in cluster_labels:
            x_data_i[c] = 1

        x_data.append(x_data_i)

        if len(tmp_imgs) == batch_size:
            outputs = predict_fn(np.concatenate(tmp_imgs, axis=0))
            y_labels.extend(outputs)
            tmp_imgs = []

    if len(tmp_imgs) > 0:
        outputs = predict_fn(np.concatenate(tmp_imgs, axis=0))
        y_labels.extend(outputs)

    return x_data, y_labels


def ridge_regressor(x_data, y_labels, softmax):
    clf = Ridge()
    clf.fit(x_data, y_labels)
    num_classes = clf.coef_.shape[0]
    global_weights_all_labels = {}

    if len(y_labels) / num_classes < 3:
        print("Warning: The test samples in the dataset is limited.\n "
              "NormLIME may have no effect on the results.\n "
              "Try to add more test samples, or see the results of LIME.")

    # clf.coef_ has shape of [len(np.unique(y_labels)), num_features]
    #
    for class_index in range(num_classes):
        w = clf.coef_[class_index]

        if softmax:
            w = w - np.max(w)
            exp_w = np.exp(w * 10)
            w = exp_w / np.sum(exp_w)

        global_weights_all_labels[class_index] = {
            i: wi
            for i, wi in enumerate(w)
        }

    return global_weights_all_labels


def precompute_global_prior(test_set_file_list,
                            predict_fn,
                            batch_size,
                            gp_method="ridge",
                            softmax=False):

    x_data, y_labels = data_labels(test_set_file_list, predict_fn, batch_size)

    if gp_method.lower() == 'ridge':
        global_weights_all_labels = ridge_regressor(x_data, y_labels, softmax)
    else:
        return None

    return global_weights_all_labels


def use_fast_normlime_as_prior(image_show, segments, label_index,
                               global_weights):
    assert isinstance(global_weights, dict)

    _, h_pre_models_kmeans = get_pre_models()

    try:
        kmeans_model = load_pickle_file(h_pre_models_kmeans)
    except:
        raise ValueError(
            "NormLIME needs the KMeans model, where we provided a default one in "
            "pre_models/kmeans_model.pkl.")

    fextractor = FeatureExtractor()
    feature = fextractor.forward(image_show).transpose((1, 2, 0))
    X = extract_superpixel_features(feature, segments)

    try:
        cluster_labels = kmeans_model.predict(X)
    except AttributeError:
        from sklearn.metrics import pairwise_distances_argmin_min
        cluster_labels, _ = pairwise_distances_argmin_min(
            X, kmeans_model.cluster_centers_)

    cluster_weights = global_weights.get(label_index, {})
    local_weights = [cluster_weights.get(k, 0.0) for k in cluster_labels]

    return local_weights
