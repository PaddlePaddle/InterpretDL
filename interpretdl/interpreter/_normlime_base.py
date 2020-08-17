import numpy as np
import os, sys

from ..common.paddle_utils import FeatureExtractor, extract_superpixel_features, get_pre_models
from ..data_processor.readers import load_pickle_file
from .lime import LIMECVInterpreter, LIMENLPInterpreter


class NormLIMEBase(object):
    def __init__(self,
                 image_paths,
                 predict_fn,
                 temp_data_file='all_lime_weights.npz',
                 batch_size=50,
                 num_samples=2000):

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

        self.image_paths = image_paths
        self.predict_fn = predict_fn
        self.batch_size = batch_size
        self.num_samples = num_samples

    def _get_lime_weights(self, data, auto_save=True, **kwargs):
        if 'unk_id' in kwargs:
            dict_key = '_'.join(str(i) for i in data)
        else:
            dict_key = data

        if dict_key in self.all_lime_weights:
            return

        lime_weights = self.lime_interpreter.interpret(data, **kwargs)

        if 'unk_id' in kwargs:
            self.all_lime_weights[dict_key] = {'lime_weights': lime_weights, }

        else:

            sp_seg = self.lime_interpreter.lime_intermediate_results[
                'segmentation']
            data_instance = self.lime_interpreter.lime_intermediate_results[
                'input']

            self.all_lime_weights[dict_key] = {
                'lime_weights': lime_weights,
                'segmentation': sp_seg,
                'input': data_instance
            }

        if auto_save:
            np.savez(self.filepath_to_save, **self.all_lime_weights)
            # load: dict(np.load(filepath_to_load, allow_pickle=True))

        return

    def compute_normlime(self, save_path='normlime_weights.npy'):
        self.lime_interpreter = LIMECVInterpreter(None, None)
        self.lime_interpreter._paddle_prepare(self.predict_fn)
        _, h_pre_models_kmeans = get_pre_models()
        kmeans_model = load_pickle_file(h_pre_models_kmeans)

        last_len = 0
        # compute lime weights and put in self.all_lime_weights
        for i, image_path in enumerate(self.image_paths):
            sys.stdout.write('\r')
            out_str = f"computing {image_path.split('/')[-1]}, {i+1}/{len(self.image_paths)}"
            sys.stdout.write(out_str + ' ' * max(last_len - len(out_str), 0))
            last_len = len(out_str)
            sys.stdout.flush()

            #print(
            #    f"computing {image_path.split('/')[-1]}, {i}/{len(self.image_paths)}"
            #)
            self._get_lime_weights(
                image_path,
                auto_save=(i % 10 == 0),
                num_samples=self.num_samples,
                batch_size=self.batch_size)
        print('\n')
        np.savez(self.filepath_to_save, **self.all_lime_weights)

        # convert superpixel indexes to cluster indexes.
        normlime_weights_all_labels = {}
        for i, image_path in enumerate(self.image_paths):
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

    def compute_normlime_text(self, unk_id, save_path='normlime_weights.npy'):
        self.lime_interpreter = LIMENLPInterpreter(None, None)
        self.lime_interpreter._paddle_prepare(self.predict_fn)
        data = self.image_paths
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = data
        else:
            seq_lens = data.recursive_sequence_lengths()[0]
            data = np.array(data)
            self.data = []
            start = 0
            for l in seq_lens:
                self.data.append(data[start:start + l])
                start += l

        # compute lime weights and put in self.all_lime_weights
        for i in range(len(self.data)):
            sys.stdout.write('\r')
            sys.stdout.write(f"computing {i+1}/{len(self.data)}")
            sys.stdout.flush()
            self._get_lime_weights(
                np.array(self.data[i]),
                auto_save=(i % 10 == 0),
                unk_id=unk_id,
                num_samples=self.num_samples,
                batch_size=self.batch_size)
        print('\n')
        np.savez(self.filepath_to_save, **self.all_lime_weights)
        normlime_weights_all_labels = {}
        unique_ids = np.unique(data)
        for i in range(len(self.data)):
            data_instance = self.data[i]
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
