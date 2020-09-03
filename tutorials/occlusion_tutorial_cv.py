import sys
sys.path.append('..')
from interpretdl.data_processor.visualizer import visualize_grayscale

from tutorials.assets.resnet import ResNet50
from interpretdl import OcclusionInterpreter
from PIL import Image
import numpy as np


def occlusion_example():
    def paddle_model(image_input):
        import paddle.fluid as fluid

        class_num = 1000
        model = ResNet50()
        logits = model.net(input=image_input, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    # The model can be downloaded from
    # http://paddle-imagenet-models-name.bj.bcebos.com/ResNet150_pretrained.tar
    # More pretrained models can be found in
    # https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification
    img_path = 'assets/catdog.png'

    oc = OcclusionInterpreter(paddle_model, "assets/ResNet50_pretrained")
    attributions = oc.interpret(
        img_path,
        sliding_window_shapes=(1, 50, 50),
        labels=None,
        strides=(1, 30, 30),
        baselines=None,
        perturbations_per_eval=5,
        visual=True,
        save_path='assets/occlusion_gray.jpg')


if __name__ == '__main__':
    occlusion_example()
