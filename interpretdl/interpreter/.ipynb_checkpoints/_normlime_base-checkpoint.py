import numpy as np
import os, sys
from tqdm import tqdm

from ..common.paddle_utils import FeatureExtractor, extract_superpixel_features, get_pre_models
from ..data_processor.readers import load_pickle_file
from .lime import LIMECVInterpreter, LIMENLPInterpreter


class NormLIMECVInterpreter(LIMECVInterpreter):
    """
    NormLIME Interpreter for CV tasks.

    More details regarding the NormLIME method can be found in the original paper:
    https://arxiv.org/abs/1909.04200
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 model_input_shape=[3, 224, 224],
                 use_cuda=True,
                 temp_data_file='all_lime_weights.npz'):
        """
        Initialize the NormLIMECVInterpreter.


        :param paddle_model: A user-defined function that gives access to model predictions.
                    It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.
        :type paddle_model: callable
        :param trained_model_path: The pretrained model directory.
        :type trained_model_path: str
        :param model_input_shape: The input shape of the model. Default: [3, 224, 224]
        :type model_input_shape: list, optional
        :param use_cuda: Whether or not to use cuda. Default: True
        :type use_cuda: bool, optional
        :param temp_data_file: The .npz file to save/load the dictionary where key is image path and value is another dictionary with lime weights, segmentation and input. Default: 'all_lime_weights.npz'
        :type temp_data_file: str, optional
        """
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
            data, num_samples=num_samples, batch_size=batch_size, visual=False)

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
                  num_samples=1000,
                  batch_size=50,
                  save_path='normlime_weights.npy'):
        """
        Main function of the interpreter.

        Args:
            image_paths (list of strs): A list of image filepaths.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            save_path (str, optional): The .npy path to save the normlime weights. It is a dictionary where the key is label and value is segmentation ids with their importance. Default: 'normlime_weights.npy'

        :return: NormLIME weights: {label_i: weights on features}
        :rtype: dict

        Example::

            def paddle_model(image_input):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs

            # The model can be downloaded from
            # http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar
            # More pretrained models can be found in
            # https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification

            # 10 images are used here for example, but more images should be used.
            dataset_dir = "assets"
            image_paths = sorted(glob.glob(dataset_dir + "/*.png"))
            image_paths = image_paths[:10]

            normlime = it.NormLIMECVInterpreter(paddle_model,
                                                "assets/ResNet50_pretrained")

            # this can be very slow.
            normlime.interpret(image_paths, num_samples=2000, batch_size=50)


        """
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
    """
    NormLIME Interpreter for NLP tasks.

    More details regarding the NormLIME method can be found in the original paper:
    https://arxiv.org/abs/1909.04200
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 temp_data_file='all_lime_weights.npz'):
        """
        Initialize the NormLIMENLPInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                    It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            temp_data_file (str, optinal): The .npz file to save/load the dictionary where key is word ids joined by '-' and value is another dictionary with lime weights. Default: 'all_lime_weights.npz'
        """
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
                          dict_key,
                          preprocess_fn,
                          num_samples,
                          batch_size,
                          unk_id,
                          pad_id,
                          lod_levels,
                          auto_save=True):

        #dict_key = '_'.join(str(i) for i in data)
        #dict_key = data
        if dict_key in self.all_lime_weights:
            return

        lime_weights = self.lime_interpret(
            data,
            preprocess_fn=preprocess_fn,
            unk_id=unk_id,
            pad_id=pad_id,
            num_samples=num_samples,
            lod_levels=lod_levels,
            batch_size=batch_size)

        self.all_lime_weights[dict_key] = {'lime_weights': lime_weights, }

        if auto_save:
            np.savez(self.filepath_to_save, **self.all_lime_weights)
            # load: dict(np.load(filepath_to_load, allow_pickle=True))

        return

    def interpret(self,
                  data,
                  preprocess_fn,
                  num_samples,
                  batch_size,
                  unk_id,
                  pad_id=None,
                  lod_levels=None,
                  save_path='normlime_weights.npy'):
        """
        Main function of the interpreter.

        Args:
            data (str): The raw string for analysis.
            preprocess_fn (Callable): A user-defined function that input raw string and outputs the a tuple of inputs to feed into the NLP model.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. Default: None.
            lod_levels (list or tuple or numpy.ndarray or None, optional): The lod levels for model inputs. It should have the length equal to number of outputs given by preprocess_fn.
                                                        If None, lod levels are all zeros. Default: None.
            save_path (str, optional): The .npy path to save the normlime weights. It is a dictionary where the key is label and value is segmentation ids with their importance. Default: 'normlime_weights.npy'

        :return: NormLIME weights: {label_i: weights on words}
        :rtype: dict

        Example::

            import interpretdl as it
            from assets.bilstm import bilstm
            import io
            import paddle.fluid as fluid

            def paddle_model(data, seq_len):
                probs = bilstm(data, seq_len, None, DICT_DIM, is_prediction=True)
                return probs

            MAX_SEQ_LEN = 256
            MODEL_PATH = "assets/senta_model/bilstm_model/"
            VOCAB_PATH = os.path.join(MODEL_PATH, "word_dict.txt")
            PARAMS_PATH = os.path.join(MODEL_PATH, "params")
            DATA_PATH = "assets/senta_data/test.tsv"

            def preprocess_fn(data):
                word_ids = []
                sub_word_ids = [
                    word_dict.get(d, word_dict['<unk>']) for d in data.split()
                ]
                seq_lens = [len(sub_word_ids)]
                if len(sub_word_ids) < MAX_SEQ_LEN:
                    sub_word_ids += [0] * (MAX_SEQ_LEN - len(sub_word_ids))
                word_ids.append(sub_word_ids[:MAX_SEQ_LEN])
                return word_ids, seq_lens

            def load_vocab(file_path):
                vocab = {}
                with io.open(file_path, 'r', encoding='utf8') as f:
                    wid = 0
                    for line in f:
                        if line.strip() not in vocab:
                            vocab[line.strip()] = wid
                            wid += 1
                vocab["<unk>"] = len(vocab)
                return vocab

            DICT_DIM = 1256606

            word_dict = load_vocab(VOCAB_PATH)
            unk_id = word_dict[""]  #["<unk>"]

            pad_id = 0
            data = []
            max_len = 512
            with io.open(DATA_PATH, "r", encoding='utf8') as fin:
                for line in fin:
                    if line.startswith('text_a'):
                        continue
                    cols = line.strip().split("\t")
                    if len(cols) != 2:
                        sys.stderr.write("[NOTICE] Error Format Line!")
                        continue
                    data.append(cols[0])
            print('total of %d sentences' % len(data))

            normlime = it.NormLIMENLPInterpreter(
                paddle_model, PARAMS_PATH, temp_data_file='all_lime_weights_nlp.npz')

            normlime_weights = normlime.interpret(
                data,
                preprocess_fn,
                unk_id=unk_id,
                pad_id=0,
                num_samples=500,
                batch_size=50)

            id2word = dict(zip(word_dict.values(), word_dict.keys()))
            for label in normlime_weights:
                print(label)
                temp = {
                    id2word[wid]: normlime_weights[label][wid]
                    for wid in normlime_weights[label]
                }
                W = [(word, weight[0], weight[1]) for word, weight in temp.items()]
                print(sorted(W, key=lambda x: -x[1])[:15])


        """

        # compute lime weights and put in self.all_lime_weights
        for i in tqdm(range(len(data))):
            self._get_lime_weights(
                data[i],
                preprocess_fn=preprocess_fn,
                dict_key=str(i),
                unk_id=unk_id,
                pad_id=pad_id,
                num_samples=num_samples,
                batch_size=batch_size,
                lod_levels=lod_levels,
                auto_save=(i % 10) == 0)

        np.savez(self.filepath_to_save, **self.all_lime_weights)

        normlime_weights_all_labels = {}
        for i in range(len(data)):
            data_instance = data[i]
            temp = self.all_lime_weights[str(i)]
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
