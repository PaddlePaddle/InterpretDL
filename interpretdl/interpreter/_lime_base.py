"""
Copyright (c) 2016, Marco Tulio Correia Ribeiro
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
The code in this file (_lime_base.py) is largely simplified and modified from https://github.com/marcotcr/lime.
"""

import numpy as np
import sklearn
import sklearn.preprocessing
from skimage.color import gray2rgb
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize

import copy
from functools import partial
from skimage.segmentation import quickshift
from skimage.measure import regionprops


class LimeBase(object):
    """
    Class for learning a locally linear sparse model from perturbed data
    """
    def __init__(self,
                 kernel_width=0.25,
                 kernel=None,
                 verbose=False,
                 random_state=None):
        """Init function

        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def _fitting_data(self,
                      neighborhood_data,
                      neighborhood_labels,
                      distances,
                      label,
                      model_regressor=None):

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = np.array(range(neighborhood_data.shape[1]))

        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data,
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data,
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)

    def _data_labels(self,
                     image,
                     fudged_image,
                     segments,
                     classifier_fn,
                     num_samples,
                     batch_size):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features) \
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))

                if isinstance(preds, list):
                    preds = preds[0]

                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)

    def _fitting_data_with_prior(self,
                                 neighborhood_data,
                                 neighborhood_labels,
                                 distances,
                                 label,
                                 prior=None,
                                 prior_scale=1.0,
                                 reg_force=1.0
                                 ):

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = np.array(range(neighborhood_data.shape[1]))

        # use this regressor just for creating an instance of estimator.
        model_regressor = Ridge(alpha=0, fit_intercept=True,
                                random_state=self.random_state)
        easy_model = model_regressor

        X = np.float32(neighborhood_data[:, used_features])
        y = labels_column

        # pre-process
        X_offset = np.average(X, axis=0, weights=weights)
        X -= X_offset
        X, X_scale = normalize(X, axis=0, copy=False, return_norm=True)

        y_offset = np.average(y, axis=0, weights=weights)
        y -= y_offset
        y = np.reshape(y, (-1, 1))
        y, y_scale = normalize(y, axis=0, copy=False, return_norm=True)
        y = np.reshape(y, (-1))

        if weights is not None:
            # rescale by weights
            sample_weights = np.diag(np.sqrt(weights))
            X = np.dot(sample_weights, X)
            y = np.dot(sample_weights, y)

        if prior is not None:
            w0 = np.zeros(len(used_features)) if prior is None else prior
            w0 = np.array(w0)
            w0 = w0 / np.sum(np.abs(w0))
        else:
            w0 = np.zeros(len(used_features))

        I = np.identity(len(used_features))

        beta = reg_force
        tau = prior_scale
        w = np.dot(np.linalg.inv(np.dot(X.T, X) + beta * I), np.dot(X.T, y) + beta * tau * w0)

        easy_model.coef_ = w
        easy_model.intercept_ = 0

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Right:', neighborhood_labels[0, label])
        return (
            easy_model.intercept_,
            sorted(zip(used_features, easy_model.coef_),
                   key=lambda x: np.abs(x[1]), reverse=True),
            0,
            0
        )

    def interpret_instance(self, image, classifier_fn,
                           interpret_labels=(1,),
                           num_samples=1000,
                           batch_size=10,
                           hide_color=None,
                           distance_metric='cosine',
                           model_regressor=None,
                           segments=None,
                           prior=None,
                           reg_force=1.0
                           ):
        """
        Generates interpretations for a prediction.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)

        if segments is None:
            segments = compute_segments(image)

        self.segments = segments

        fudged_image = image.copy()
        if hide_color is None:
            # if no hide_color, use the mean
            for x in np.unique(segments):
                mx = np.mean(image[segments == x], axis=0)
                fudged_image[segments == x] = mx
        elif hide_color == 'avg_from_neighbor':
            from scipy.spatial.distance import cdist

            n_features = np.unique(segments).shape[0]
            regions = regionprops(segments + 1)
            centroids = np.zeros((n_features, 2))
            for i, x in enumerate(regions):
                centroids[i] = np.array(x.centroid)

            d = cdist(centroids, centroids, 'sqeuclidean')

            for x in np.unique(segments):
                # print(np.argmin(d[x]))
                a = [image[segments == i] for i in np.argsort(d[x])[1:6]]
                mx = np.mean(np.concatenate(a), axis=0)
                fudged_image[segments == x] = mx
        else:
            fudged_image[:] = 0

        data, labels = self._data_labels(image, fudged_image, segments,
                                         classifier_fn, num_samples,
                                         batch_size=batch_size)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        lime_weights = {}
        for l in interpret_labels:
            if prior is None:
                (_, lime_weights[l], _, _) = self._fitting_data(data, labels, distances, l, model_regressor)
            else:
                (_, lime_weights[l], _, _) = self._fitting_data_with_prior(data, labels, distances, l,
                                                                           prior, reg_force=reg_force)

        return lime_weights


def compute_segments(image):
    assert len(image.shape) == 3 and image.shape[-1] == 3, "Shape Error when computing superpixels."
    segments = quickshift(image, sigma=1)
    return segments