from assets.resnet import ResNet50
import paddle.fluid as fluid
import paddle
import numpy as np
import sys
sys.path.append('..')
import interpretdl as it
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import visualize_overlay
from PIL import Image


def int_grad_example():
    def paddle_model(data):
        class_num = 1000
        model = ResNet50()
        logits = model.net(input=data, class_dim=class_num)
        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    img_path = 'assets/fireboat.png'

    #https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification
    ig = it.IntGradCVInterpreter(paddle_model, "assets/ResNet50_pretrained",
                                 True)
    gradients = ig.interpret(
        img_path,
        labels=None,
        baselines='random',
        steps=50,
        num_random_trials=2,
        visual=True,
        save_path='assets/ig_test.jpg')


if __name__ == '__main__':
    int_grad_example()
