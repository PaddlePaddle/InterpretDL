from assets.resnet import ResNet50

import sys
sys.path.append('..')
#from interpretdl.interpreter.score_cam import ScoreCAMInterpreter
import interpretdl as it
from interpretdl.data_processor.readers import preprocess_image, read_image


def grad_cam_example():
    def paddle_model(image_input):
        import paddle.fluid as fluid
        class_num = 1000

        model = ResNet50()
        logits = model.net(input=image_input, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    img_path = 'assets/catdog.png'
    scorecam = it.ScoreCAMInterpreter(paddle_model,
                                      "assets/ResNet50_pretrained", True)
    scorecam.interpret(
        img_path,
        'res5c.add.output.5.tmp_0',
        labels=None,
        visual=True,
        save_path='assets/sc_test.jpg')


if __name__ == '__main__':
    grad_cam_example()
