from assets.bilstm import bilstm_net_emb
import paddle.fluid as fluid
import paddle
import numpy as np
import sys, os
sys.path.append('..')
import interpretdl as it
from interpretdl.data_processor.visualizer import VisualizationTextRecord, visualize_text


def nlp_example_bilstm():
    #Dataset: https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
    #Pretrained Model: https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz
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

    def paddle_model(data, alpha):
        dict_dim = 1256606
        emb_dim = 128
        # embedding layer
        emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
        emb *= alpha
        probs = bilstm_net_emb(emb, None, None, dict_dim, is_prediction=True)
        return emb, probs

    MODEL_PATH = "assets/senta_model/bilstm_model/"
    PARAMS_PATH = os.path.join(MODEL_PATH, "params")
    VOCAB_PATH = os.path.join(MODEL_PATH, "word_dict.txt")

    ig = it.IntGradNLPInterpreter(paddle_model, PARAMS_PATH, True)

    word_dict = load_vocab(VOCAB_PATH)
    unk_id = word_dict["<unk>"]
    reviews = [[
        '交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'
    ]]

    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, unk_id) for words in c])
    base_shape = [[len(c) for c in lod]]
    lod = np.array(sum(lod, []), dtype=np.int64)
    data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())

    pred_labels, pred_probs, avg_gradients = ig.interpret(
        data, label=None, steps=50, return_pred=True, visual=True)

    sum_gradients = np.sum(avg_gradients, axis=1).tolist()
    lod = data.lod()

    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(
            list(zip(reviews[i], sum_gradients[lod[0][i]:lod[0][i + 1]])))

    print(new_array)

    true_labels = [1]
    recs = []
    for i, l in enumerate(new_array):
        words = [t[0] for t in l]
        word_importances = [t[1] for t in l]
        word_importances = np.array(word_importances) / np.linalg.norm(
            word_importances)
        pred_label = pred_labels[i]
        pred_prob = pred_probs[i]
        true_label = true_labels[0]
        interp_class = pred_label
        if interp_class == 0:
            word_importances = -word_importances
        print(words, word_importances)
        recs.append(
            VisualizationTextRecord(words, word_importances, true_label,
                                    pred_label, pred_prob, interp_class))

    visualize_text(recs)


def nlp_example_conv():
    # Modified from https://www.paddlepaddle.org.cn/documentation/docs/en/user_guides/nlp_case/understand_sentiment/README.html
    def convolution_net(emb, input_dim, class_dim, emb_dim, hid_dim):
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
    BATCH_SIZE = 128
    word_dict = paddle.dataset.imdb.word_dict()

    def paddle_model(data, alpha):
        emb = fluid.embedding(
            input=data, size=[len(word_dict), EMB_DIM], is_sparse=True)
        emb = emb * alpha
        probs = convolution_net(emb,
                                len(word_dict), CLASS_DIM, EMB_DIM, HID_DIM)
        return emb, probs

    ig = it.IntGradNLPInterpreter(
        paddle_model,
        "assets/sent_persistables",  #Training based on https://www.paddlepaddle.org.cn/documentation/docs/en/user_guides/nlp_case/understand_sentiment/README.html
        True)

    reviews_str = [
        b'read the book forget the movie', b'this is a great movie',
        b'this is very bad'
    ]
    reviews = [c.split() for c in reviews_str]
    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])
    base_shape = [[len(c) for c in lod]]
    lod = np.array(sum(lod, []), dtype=np.int64)
    data = fluid.create_lod_tensor(lod, base_shape, fluid.CUDAPlace(0))

    avg_gradients = ig.interpret(
        data, label=None, steps=50, visual=True, save_path='ig_test.jpg')

    sum_gradients = np.sum(avg_gradients, axis=1).tolist()
    lod = data.lod()

    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(
            dict(zip(reviews[i], sum_gradients[lod[0][i]:lod[0][i + 1]])))

    print(new_array)


if __name__ == '__main__':
    target = sys.argv[1:]
    if 'conv' in target:
        nlp_example_conv()
    elif 'bilstm' in target:
        nlp_example_bilstm()
