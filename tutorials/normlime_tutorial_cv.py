import glob
import sys
import numpy as np
sys.path.append('..')

import interpretdl as it
from interpretdl.interpreter._normlime_base import NormLIMEBase
from assets.resnet import ResNet50


def normlime_example():
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
    lime._paddle_prepare()

    # 10 images are used here for example, but more images should be used.
    dataset_dir = "assets"
    image_paths = sorted(glob.glob(dataset_dir + "/*.png"))
    image_paths = image_paths[:10]

    normlime = NormLIMEBase(image_paths, lime.predict_fn)

    # this can be very slow.
    normlime.compute_normlime()


if __name__ == '__main__':
    normlime_example()
