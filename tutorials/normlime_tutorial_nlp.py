import glob
import sys, os
import numpy as np
sys.path.append('..')

import interpretdl as it


def nlp_example(dataset=True):
    from assets.bilstm import bilstm_net
    import io
    import paddle.fluid as fluid

    def load_dataset(fp):
        pad_id = 0
        all_data = []
        max_len = 512
        # prepare the dataset as a list of lists
        with io.open(fp, "r", encoding='utf8') as fin:
            for line in fin:
                if line.startswith('text_a'):
                    continue
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                wids = [
                    word_dict[x] if x in word_dict else unk_id
                    for x in cols[0][:200].split(" ")
                ]
                seq_len = len(wids)
                if seq_len < max_len:
                    wids = wids[:max_len]
                all_data.append(wids)
        print('total of %d sentences' % len(all_data))
        return all_data

    def load_vocab(file_path):
        # construct a word to word id mapping
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
    DATA_PATH = "assets/senta_data/test.tsv"
    MODEL_PATH = "assets/senta_model/bilstm_model"

    def paddle_model(data):
        probs = bilstm_net(data, None, None, DICT_DIM, is_prediction=True)
        return probs

    word_dict = load_vocab(os.path.join(MODEL_PATH, "word_dict.txt"))
    # the word id that replace occluded word, typical choices include "", <unk>, and <pad>
    unk_id = word_dict[""]  #["<unk>"]

    if dataset:
        # use the senta_data dataset, which can be downloaded from https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
        # it contains 1200 reviews
        all_data = load_dataset(DATA_PATH)
    else:
        # if not using the senta_data dataset, make some reviews on our own
        reviews = [[
            '交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'
        ], ['交通', '一般', '；', '环境', '很差', '；', '服务态度', '很差', '房间', '较小']]

        all_data = []
        for c in reviews:
            all_data.append([word_dict.get(words, unk_id) for words in c])
    # initialize the interpreter
    normlime = it.NormLIMENLPInterpreter(
        paddle_model,
        os.path.join(MODEL_PATH, "params"),
        temp_data_file='all_lime_weights_nlp.npz')
    # begin interpretation
    normlime_weights = normlime.interpret(
        all_data, unk_id, num_samples=500, batch_size=50)

    # construct a word id to word mapping
    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for label in normlime_weights:
        # print out the label and word importances
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
