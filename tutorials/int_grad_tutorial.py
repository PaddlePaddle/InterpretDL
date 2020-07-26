from assets.resnet import ResNet50
from assets.bilstm import bilstm_net
import paddle.fluid as fluid
import numpy as np
import sys
sys.path.append('..')
from interpretdl.interpreter.integrated_gradients import IntGradInterpreter
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import visualize_ig
from PIL import Image


def int_grad_example():
    def paddle_model(data, alpha, baseline):

        class_num = 1000
        image_input = baseline + alpha * data
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
        num_random_trials=1,
        visual=True,
        save_path='ig_test.jpg')


def nlp_example():
    def paddle_model(data, alpha, baseline):
        seq_len = fluid.layers.fill_constant(
            shape=[1], value=12, dtype='int64')
        emb, probs = bilstm_net(
            data, seq_len, None, 1256606, is_prediction=True, alpha=alpha)
        return emb, probs

    #https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
    ig = IntGradInterpreter(
        paddle_model,
        "assets/senta_model/bilstm_model/params",
        2,
        True,
        model_input_shape=[256])
    data = np.array([[
        1251507, 595755, 1106205, 860907, 1134818, 1106205, 810335, 1134818,
        4779, 4779, 672177, 280917
    ] + [0] * 244])
    avg_gradients = ig.interpret(
        data,
        label=None,
        baseline='random',
        steps=50,
        num_random_trials=1,
        visual=True,
        save_path='ig_test.jpg')
    print(np.sum(avg_gradients, axis=1))


if __name__ == '__main__':
    target = sys.argv[1:]
    if 'nlp' in target:
        nlp_example()
    else:
        int_grad_example()
