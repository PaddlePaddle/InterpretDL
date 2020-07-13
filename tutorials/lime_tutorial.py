import sys
sys.path.append('..')

from tutorials.assets.resnet import ResNet101
from interpretdl import LIMEInterpreter


def lime_example():
    def predict_fn(image_input):
        import paddle.fluid as fluid

        class_num = 1000
        model = ResNet101()
        logits = model.net(input=image_input, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    # The model can be downloaded from
    # http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar
    # More pretrained models can be found in
    # https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification
    lime = LIMEInterpreter(predict_fn, "assets/ResNet101_pretrained")
    lime.interpret('assets/catdog.png', num_samples=1000, batch_size=100, save_path='assets/catdog_lime.png',)


if __name__ == '__main__':
    lime_example()