import glob
import sys, os
import numpy as np
sys.path.append('..')

import interpretdl as it


def nlp_example(dataset=True):
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

    if dataset:
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
    else:
        reviews = [
            '交通 方便 ；环境 很好 ；服务态度 很好 房间 较小', '交通 一般 ；环境 很差 ；服务态度 很差 房间 较小'
        ]
        data = reviews

    normlime = it.NormLIMENLPInterpreter(
        paddle_model, PARAMS_PATH, temp_data_file='all_lime_weights_nlp.npz')

    normlime_weights = normlime.interpret(
        data,
        preprocess_fn,
        unk_id=unk_id,
        pad_id=0,
        num_samples=500,
        batch_size=50)

    #print(normlime_weights)
    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for label in normlime_weights:
        print(label)
        temp = {
            id2word[wid]: normlime_weights[label][wid]
            for wid in normlime_weights[label]
        }
        W = [(word, weight[0], weight[1]) for word, weight in temp.items()]
        print(sorted(W, key=lambda x: -x[1])[:15])


if __name__ == '__main__':
    targets = sys.argv[1:]
    if 'nlp_mini' in targets:
        nlp_example(dataset=False)
    else:
        nlp_example()
