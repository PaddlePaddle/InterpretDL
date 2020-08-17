import numpy as np
import os, sys
from tqdm import tqdm

from ..common.paddle_utils import FeatureExtractor, extract_superpixel_features, get_pre_models
from ..data_processor.readers import load_pickle_file
from .lime import LIMECVInterpreter, LIMENLPInterpreter


class NormLIMECVInterpreter(LIMECVInterpreter):
    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 model_input_shape=[3, 224, 224],
                 use_cuda=True,
                 temp_data_file='all_lime_weights.npz'):
        LIMECVInterpreter.__init__(self, paddle_model, trained_model_path,
                                   model_input_shape, use_cuda)
        self.lime_interpret = super().interpret

        if temp_data_file.endswith('.npz'):
            self.filepath_to_save = temp_data_file
        else:
            self.filepath_to_save = temp_data_file + '.npz'

        if os.path.exists(self.filepath_to_save):
            self.all_lime_weights = dict(
                np.load(
                    self.filepath_to_save, allow_pickle=True))
        else:
            self.all_lime_weights = {}

    def _get_lime_weights(self, data, num_samples, batch_size, auto_save=True):
        if data in self.all_lime_weights:
            return
        lime_weights = self.lime_interpret(
            data, num_samples=num_samples, batch_size=batch_size)

        sp_seg = self.lime_intermediate_results['segmentation']
        data_instance = self.lime_intermediate_results['input']

        self.all_lime_weights[data] = {
            'lime_weights': lime_weights,
            'segmentation': sp_seg,
            'input': data_instance
        }

        if auto_save:
            np.savez(self.filepath_to_save, **self.all_lime_weights)
            # load: dict(np.load(filepath_to_load, allow_pickle=true))

        return

    def interpret(self,
                  image_paths,
                  num_samples=2000,
                  batch_size=50,
                  save_path='normlime_weights.npy'):
        #self.lime_interpreter = limecvinterpreter(none, none)
        #self.lime_interpreter._paddle_prepare(self.predict_fn)

        _, h_pre_models_kmeans = get_pre_models()
        kmeans_model = load_pickle_file(h_pre_models_kmeans)

        # compute lime weights and put in self.all_lime_weights
        for i in tqdm(range(len(image_paths))):
            image_path = image_paths[i]
            self._get_lime_weights(
                image_path, num_samples, batch_size, auto_save=(i % 10 == 0))

        np.savez(self.filepath_to_save, **self.all_lime_weights)

        # convert superpixel indexes to cluster indexes.
        normlime_weights_all_labels = {}
        for i, image_path in enumerate(image_paths):
            temp = self.all_lime_weights[image_path]
            if isinstance(temp, np.ndarray):
                temp = temp.item()

            fextractor = FeatureExtractor()
            f = fextractor.forward(temp['input'][np.newaxis, ...]).transpose(
                (1, 2, 0))

            X = extract_superpixel_features(f, temp['segmentation'])
            try:
                cluster_labels = kmeans_model.predict(
                    X)  # a list. len = number of sp.
            except AttributeError:
                from sklearn.metrics import pairwise_distances_argmin_min
                cluster_labels, _ = pairwise_distances_argmin_min(
                    X, kmeans_model.cluster_centers_)
            lime_weights = temp['lime_weights']
            pred_labels = lime_weights.keys()
            for y in pred_labels:
                normlime_weights_label_y = normlime_weights_all_labels.get(y,
                                                                           {})
                w_f_y = [abs(w[1]) for w in lime_weights[y]]
                w_f_y_l1norm = sum(w_f_y)

                for w in lime_weights[y]:
                    seg_label = w[0]
                    weight = w[1] * w[1] / w_f_y_l1norm
                    tmp = normlime_weights_label_y.get(
                        cluster_labels[seg_label], [])
                    tmp.append(weight)
                    normlime_weights_label_y[cluster_labels[seg_label]] = tmp

                normlime_weights_all_labels[y] = normlime_weights_label_y
        # compute normlime weights.
        for y in normlime_weights_all_labels:
            normlime_weights = normlime_weights_all_labels.get(y, {})
            for k in normlime_weights:
                normlime_weights[k] = sum(normlime_weights[k]) / len(
                    normlime_weights[k])

        # check normlime
        if len(normlime_weights_all_labels.keys()) < max(
                normlime_weights_all_labels.keys()) + 1:
            print(
                "\n" + \
                "Warning: !!! \n" + \
                "There are at least {} classes, ".format(max(normlime_weights_all_labels.keys()) + 1) + \
                "but the NormLIME has results of only {} classes. \n".format(len(normlime_weights_all_labels.keys())) + \
                "It may have cause unstable results in the later computation" + \
                " but can be improved by computing more test samples." + \
                "\n"
            )

        if os.path.exists(save_path):
            n = 0
            tmp = save_path.split('.npy')[0]
            while os.path.exists(f'{tmp}-{n}.npy'):
                n += 1

            np.save(f'{tmp}-{n}.npy', normlime_weights_all_labels)
        else:
            np.save(save_path, normlime_weights_all_labels)

        return normlime_weights_all_labels


class NormLIMENLPInterpreter(LIMENLPInterpreter):
    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 temp_data_file='all_lime_weights.npz'):
        LIMENLPInterpreter.__init__(self, paddle_model, trained_model_path,
                                    use_cuda)
        self.lime_interpret = super().interpret

        if temp_data_file.endswith('.npz'):
            self.filepath_to_save = temp_data_file
        else:
            self.filepath_to_save = temp_data_file + '.npz'

        if os.path.exists(self.filepath_to_save):
            self.all_lime_weights = dict(
                np.load(
                    self.filepath_to_save, allow_pickle=True))
        else:
            self.all_lime_weights = {}

    def _get_lime_weights(self,
                          data,
                          unk_id,
                          num_samples,
                          batch_size,
                          auto_save=True):

        dict_key = '_'.join(str(i) for i in data)

        if dict_key in self.all_lime_weights:
            return

        lime_weights = self.lime_interpret(
            data,
            unk_id=unk_id,
            num_samples=num_samples,
            batch_size=batch_size)

        self.all_lime_weights[dict_key] = {'lime_weights': lime_weights, }

        if auto_save:
            np.savez(self.filepath_to_save, **self.all_lime_weights)
            # load: dict(np.load(filepath_to_load, allow_pickle=True))

        return

    def interpret(self,
                  word_ids,
                  unk_id,
                  num_samples,
                  batch_size,
                  save_path='normlime_weights.npy'):
        if isinstance(word_ids, list) or isinstance(word_ids, np.ndarray):
            data = word_ids
        else:
            seq_lens = word_ids.recursive_sequence_lengths()[0]
            word_ids = np.array(word_ids)
            data = []
            start = 0
            for l in seq_lens:
                data.append(word_ids[start:start + l])
                start += l

        # compute lime weights and put in self.all_lime_weights
        for i in tqdm(range(len(data))):
            self._get_lime_weights(
                np.array(data[i]),
                unk_id,
                num_samples,
                batch_size,
                auto_save=(i % 10) == 0)

        np.savez(self.filepath_to_save, **self.all_lime_weights)

        normlime_weights_all_labels = {}
        for i in range(len(data)):
            data_instance = data[i]
            temp = self.all_lime_weights['_'.join(
                str(i) for i in data_instance)]
            if isinstance(temp, np.ndarray):
                temp = temp.item()
            lime_weights = temp['lime_weights']
            pred_labels = lime_weights.keys()
            for y in pred_labels:
                normlime_weights_label_y = normlime_weights_all_labels.get(y,
                                                                           {})
                w_f_y = [abs(w[1]) for w in lime_weights[y]]
                w_f_y_l1norm = sum(w_f_y)

                for w in lime_weights[y]:
                    word_id = w[0]
                    if w[1] > 0:
                        weight = w[1] * w[1] / w_f_y_l1norm
                        tmp = normlime_weights_label_y.get(word_id, [])
                        tmp.append(weight)
                        normlime_weights_label_y[word_id] = tmp

                normlime_weights_all_labels[y] = normlime_weights_label_y
        # compute normlime weights.
        for y in normlime_weights_all_labels:
            normlime_weights = normlime_weights_all_labels.get(y, {})
            for k in normlime_weights:
                normlime_weights[k] = (sum(normlime_weights[k]) /
                                       len(normlime_weights[k]),
                                       len(normlime_weights[k]))

        # check normlime
        if len(normlime_weights_all_labels.keys()) < max(
                normlime_weights_all_labels.keys()) + 1:
            print(
                "\n" + \
                "Warning: !!! \n" + \
                "There are at least {} classes, ".format(max(normlime_weights_all_labels.keys()) + 1) + \
                "but the NormLIME has results of only {} classes. \n".format(len(normlime_weights_all_labels.keys())) + \
                "It may have cause unstable results in the later computation" + \
                " but can be improved by computing more test samples." + \
                "\n"
            )

        if os.path.exists(save_path):
            n = 0
            tmp = save_path.split('.npy')[0]
            while os.path.exists(f'{tmp}-{n}.npy'):
                n += 1

            np.save(f'{tmp}-{n}.npy', normlime_weights_all_labels)
        else:
            np.save(save_path, normlime_weights_all_labels)

        return normlime_weights_all_labels
