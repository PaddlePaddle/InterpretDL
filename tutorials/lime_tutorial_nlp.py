import sys, os
import numpy as np
import paddle.fluid as fluid
sys.path.append('..')

import interpretdl as it


def nlp_example():
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
        review, preprocess_fn, num_samples=200, batch_size=10, unk_id=unk_id)

    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for y in lime_weights:
        print([(id2word[t[0]], t[1]) for t in lime_weights[y]])
    print(lime_weights)


if __name__ == '__main__':
    nlp_example()
