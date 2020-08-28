from assets.resnet import ResNet50
from assets.bilstm import bilstm_net
import paddle.fluid as fluid
import numpy as np
import sys
sys.path.append('..')
from interpretdl.interpreter.smooth_grad import SmoothGradInterpreter
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import visualize_grayscale
from PIL import Image
import cv2


def smooth_grad_example():
    def paddle_model(data):
        class_num = 1000
        model = ResNet50()
        logits = model.net(input=data, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    img_path = 'assets/deer.png'
    sg = SmoothGradInterpreter(paddle_model, "assets/ResNet50_pretrained")
    gradients = sg.interpret(img_path, visual=True, save_path='sg_test.jpg')

    # optional
    visualize_grayscale(gradients, save_path='sg_gray.jpg')


if __name__ == '__main__':
    smooth_grad_example()
