from assets.resnet import ResNet50
import paddle.fluid as fluid
import numpy as np
import sys
sys.path.append('..')
from interpretdl.interpreter.gradient_shap import GradShapInterpreter
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import visualize_grayscale
from PIL import Image
import cv2


def grad_shap_example():
    def predict_fn(data):

        class_num = 1000
        model = ResNet50()
        logits = model.net(input=data, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    img_path = 'assets/catdog.png'
    gs = GradShapInterpreter(predict_fn, "assets/ResNet50_pretrained", 1000,
                             True)
    gradients = gs.interpret(
        img_path,
        label=None,
        noise_amout=0.1,
        n_samples=5,
        visual=True,
        save_path='grad_shap_test.jpg')


if __name__ == '__main__':
    grad_shap_example()
