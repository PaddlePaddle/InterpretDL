import glob
import sys
sys.path.append('..')

from interpretdl.interpreter._normlime_base import NormLIMEBase
from interpretdl import LIMEInterpreter

from assets.resnet import ResNet101


def normlime_example():
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
    lime._paddle_prepare()

    # 10 images are used here for example, but more images should be used.
    dataset_dir = "assets/ILSVRC2012_images_val/val"
    image_paths = sorted(glob.glob(dataset_dir + "/*"))
    image_paths = image_paths[:10]

    normlime = NormLIMEBase(image_paths, lime.predict_fn)

    # this can be very slow.
    normlime.compute_normlime()


if __name__ == '__main__':
    normlime_example()
