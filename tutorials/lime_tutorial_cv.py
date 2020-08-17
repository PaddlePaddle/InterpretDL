import sys
import numpy as np
import paddle.fluid as fluid
sys.path.append('..')

from tutorials.assets.resnet import ResNet50
#from interpretdl import LIMECVInterpreter
import interpretdl as it


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
    lime = it.LIMECVInterpreter(paddle_model, "assets/ResNet50_pretrained")
    lime_weights = lime.interpret(
        'assets/catdog.png',
        num_samples=100,
        batch_size=10,
        save_path='catdog_lime.png')


if __name__ == '__main__':
    lime_example()
