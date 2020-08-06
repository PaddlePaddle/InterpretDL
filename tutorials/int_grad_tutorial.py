from assets.resnet import ResNet50
from assets.bilstm import bilstm_net
import paddle.fluid as fluid
import paddle
import numpy as np
import sys
sys.path.append('..')
from interpretdl.interpreter.integrated_gradients import IntGradInterpreter
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import visualize_overlay
from PIL import Image


def int_grad_example():
    def paddle_model(data, alpha, baseline):

        class_num = 1000
        image_input = baseline + alpha * (data - baseline)
        model = ResNet50()
        logits = model.net(input=image_input, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return image_input, probs

    img_path = 'assets/fireboat.png'
    #https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification
    ig = IntGradInterpreter(paddle_model, "assets/ResNet50_pretrained", True)
    gradients = ig.interpret(
        img_path,
        label=None,
        baseline='random',
        steps=50,
        num_random_trials=10,
        visual=True,
        save_path='generated/ig_test.jpg')


def nlp_example():
    def paddle_model(data, alpha, baseline):
        dict_dim = 1256606
        emb_dim = 128
        # embedding layer
        emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
        emb *= alpha
        probs = bilstm_net(emb, None, None, dict_dim, is_prediction=True)
        return emb, probs

    #Dataset: https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
    #Pretrained Model: https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz
    ig = IntGradInterpreter(
        paddle_model,
        "assets/senta_model/bilstm_model/params",
        True,
        model_input_shape=None)

    data = np.array([[
        1251507, 595755, 1106205, 860907, 1134818, 1106205, 810335, 1134818,
        4779, 4779, 672177, 280917
    ]])

    lod = np.array(sum(data.tolist(), []), dtype=np.int64)

    data = fluid.create_lod_tensor(lod, [[12]], fluid.CPUPlace())

    avg_gradients = ig.interpret(
        data,
        label=None,
        baseline='random',
        steps=50,
        num_random_trials=1,
        visual=True,
        save_path='ig_test.jpg')
    print(np.sum(avg_gradients, axis=1))


def nlp_example2():
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

    def paddle_model(data, alpha, baseline):
        emb = fluid.embedding(
            input=data, size=[len(word_dict), EMB_DIM], is_sparse=True)
        emb = emb * alpha
        probs = convolution_net(emb,
                                len(word_dict), CLASS_DIM, EMB_DIM, HID_DIM)
        return emb, probs

    ig = IntGradInterpreter(
        paddle_model,
        "assets/sent_persistables",  #Training based on https://www.paddlepaddle.org.cn/documentation/docs/en/user_guides/nlp_case/understand_sentiment/README.html
        True,
        model_input_shape=None)

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
        data,
        label=None,
        baseline='random',
        steps=50,
        num_random_trials=1,
        visual=True,
        save_path='ig_test.jpg')

    sum_gradients = np.sum(avg_gradients, axis=1).tolist()
    lod = data.lod()

    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(sum_gradients[lod[0][i]:lod[0][i + 1]])

    print(new_array)


if __name__ == '__main__':
    target = sys.argv[1:]
    if 'nlp' in target:
        nlp_example()
    elif 'nlp2' in target:
        nlp_example2()
    else:
        int_grad_example()
