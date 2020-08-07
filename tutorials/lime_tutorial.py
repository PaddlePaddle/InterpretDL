import sys
import numpy as np
import paddle.fluid as fluid
sys.path.append('..')

from tutorials.assets.resnet import ResNet50
from interpretdl import LIMEInterpreter


def lime_example():
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
    lime = LIMEInterpreter(paddle_model, "../../ResNet50_pretrained")
    lime_weights = lime.interpret(
        'assets/catdog.png',
        num_samples=100,
        batch_size=10,
        save_path='generated/catdog_lime.png')


def nlp_example():
    from assets.bilstm import bilstm_net
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

    DICT_DIM = 1256606

    def paddle_model(data):
        probs = bilstm_net(data, None, None, DICT_DIM, is_prediction=True)
        return probs

    #https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
    word_dict = load_vocab("../../senta_model/bilstm_model/word_dict.txt")
    unk_id = word_dict["<unk>"]
    lime = LIMEInterpreter(paddle_model,
                           "../../senta_model/bilstm_model/params")

    reviews = [[
        '交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'
    ]]
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, unk_id) for words in c])
    base_shape = [[len(c) for c in lod]]

    lod = np.array(sum(lod, []), dtype=np.int64)
    data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())
    lime_weights = lime.interpret(
        data, num_samples=100, batch_size=10, unk_id=unk_id)
    print(lime_weights)


def nlp_example2():
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
    BATCH_SIZE = 128
    print('Preparing word_dict...')
    word_dict = paddle.dataset.imdb.word_dict()

    def paddle_model(data):
        probs = convolution_net(data,
                                len(word_dict), CLASS_DIM, EMB_DIM, HID_DIM)
        return probs

    lime = LIMEInterpreter(paddle_model, "assets/sent_persistables")

    reviews_str = [b'read the book forget the movie']

    reviews = [c.split() for c in reviews_str]

    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    base_shape = [[len(c) for c in lod]]
    lod = np.array(sum(lod, []), dtype=np.int64)

    data = fluid.create_lod_tensor(lod, base_shape, fluid.CUDAPlace(0))
    print('Begin intepretation...')
    lime_weights = lime.interpret(
        data, num_samples=100, batch_size=10, unk_id=UNK)
    print(lime_weights)


if __name__ == '__main__':
    targets = sys.argv[1:]
    if 'nlp' in targets:
        nlp_example()
    elif 'nlp2' in targets:
        nlp_example2()
    else:
        lime_example()
