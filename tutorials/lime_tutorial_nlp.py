import sys, os
import numpy as np
import paddle.fluid as fluid
sys.path.append('..')

import interpretdl as it


def nlp_example_bilstm():
    from assets.bilstm import bilstm
    import io

    def load_vocab(file_path):
        """
        load the given vocabulary
        """
        vocab = {}
        with io.open(file_path, 'r', encoding='utf8') as f:
            wid = 0
            for line in f:
                if line.strip() not in vocab:
                    vocab[line.strip()] = wid
                    wid += 1
        vocab["<unk>"] = len(vocab)
        return vocab

    MODEL_PATH = "assets/senta_model/bilstm_model"
    VOCAB_PATH = os.path.join(MODEL_PATH, "word_dict.txt")
    PARAMS_PATH = os.path.join(MODEL_PATH, "params")
    DICT_DIM = 1256606

    def paddle_model(data, seq_len):
        probs = bilstm(data, seq_len, None, DICT_DIM, is_prediction=True)
        return probs

    MAX_SEQ_LEN = 256

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

    #https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
    word_dict = load_vocab(VOCAB_PATH)
    unk_id = word_dict[""]  #word_dict["<unk>"]
    lime = it.LIMENLPInterpreter(paddle_model, PARAMS_PATH)

    review = '交通 方便 ；环境 很好 ；服务态度 很好 房间 较小'

    lime_weights = lime.interpret(
        review,
        preprocess_fn,
        num_samples=200,
        batch_size=10,
        unk_id=unk_id,
        pad_id=0)

    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for y in lime_weights:
        print([(id2word[t[0]], t[1]) for t in lime_weights[y]])
    print(lime_weights)


def nlp_example_conv():
    import paddle

    def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
        emb = fluid.embedding(
            input=data, size=[len(word_dict), EMB_DIM], is_sparse=True)
        conv_3 = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=3,
            act="tanh",
            pool_type="sqrt")
        conv_4 = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=4,
            act="tanh",
            pool_type="sqrt")
        prediction = fluid.layers.fc(input=[conv_3, conv_4],
                                     size=class_dim,
                                     act="softmax")
        return prediction

    CLASS_DIM = 2
    EMB_DIM = 128
    HID_DIM = 512
    PARAMS_PATH = "assets/InterpretDL/tutorials/assets/sent_persistables"
    print('preparing word_dict...')
    word_dict = paddle.dataset.imdb.word_dict()

    def paddle_model(data):
        probs = convolution_net(data,
                                len(word_dict), CLASS_DIM, EMB_DIM, HID_DIM)
        return probs

    lime = it.LIMENLPInterpreter(paddle_model, PARAMS_PATH)

    data = b'read the book forget the movie'

    def preprocess_fn(data):
        reviews = [[
            word_dict.get(w, word_dict['<unk>']) for w in data.split()
        ]]
        return reviews

    UNK = word_dict['<unk>']

    lime_weights = lime.interpret(
        data,
        preprocess_fn,
        num_samples=200,
        batch_size=20,
        unk_id=UNK,
        pad_id=None,
        lod_levels=[1])
    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for y in lime_weights:
        print([(id2word[t[0]], t[1]) for t in lime_weights[y]])
    print(lime_weights)


def nlp_example_ernie():
    from assets.bilstm import bilstm
    import io
    sys.path.append("../../../models/PaddleNLP/shared_modules/")
    from preprocess.ernie import task_reader, tokenization
    from collections import namedtuple
    from models.classification.nets import ernie_base_net
    from models.representation.ernie import ErnieConfig, ernie_encoder

    MODEL_PATH = 'assets/senta_model/ernie_pretrain_model'
    VOCAB_PATH = os.path.join(MODEL_PATH, 'vocab.txt')
    PARAMS_PATH = 'assets/senta_model/ernie_trained/step_1000'
    ERNIE_CONFIG_PATH = os.path.join(MODEL_PATH, 'ernie_config.json')
    MAX_SEQ_LEN = 256

    ernie_config = ErnieConfig(ERNIE_CONFIG_PATH)

    reader = task_reader.ClassifyReader(
        vocab_path=VOCAB_PATH, max_seq_len=MAX_SEQ_LEN, do_lower_case=True)

    def paddle_model(src_ids, sent_ids, pos_ids, input_mask, labels, seq_lens):
        ernie_inputs = {
            "src_ids": src_ids,
            "sent_ids": sent_ids,
            "pos_ids": pos_ids,
            "input_mask": input_mask,
            "seq_lens": seq_lens
        }

        embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)
        ce_loss, probs = ernie_base_net(embeddings["sentence_embeddings"],
                                        labels, 2)

        return probs

    def preprocess_fn(data):
        Example = namedtuple('Example', ['text_a', 'label'])
        example = Example(data.replace(' ', ''), 0)
        for return_list in reader._prepare_batch_data([example], 1):
            return tuple(return_list)

    #print(preprocess_fn('交通 方便 ；环境 很好 ；服务态度 很好 房间 较小'))
    word_dict = tokenization.load_vocab(VOCAB_PATH)
    unk_id = word_dict["[UNK]"]
    lime = it.LIMENLPInterpreter(paddle_model, PARAMS_PATH)
    review = '交通 方便 ；环境 很好 ；服务态度 很好 房间 较小'

    lime_weights = lime.interpret(
        review,
        preprocess_fn,
        num_samples=500,
        batch_size=50,
        unk_id=unk_id,
        pad_id=0)
    print(lime_weights)
    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for y in lime_weights:
        print([(id2word[t[0]], t[1]) for t in lime_weights[y]])
    print(lime_weights)


if __name__ == '__main__':
    targets = sys.argv[1:]
    if 'ernie' in targets:
        nlp_example_ernie()
    elif 'bilstm' in targets:
        nlp_example_bilstm()
    elif 'conv' in targets:
        nlp_example_conv()
