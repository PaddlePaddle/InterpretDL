import paddle
import unittest
import numpy as np
import interpretdl as it
from paddle.utils.download import get_weights_path_from_url

from tutorials.assets.vision_transformer import ViT_base_patch16_224
from tests.utils import assert_arrays_almost_equal


class TestRollout(unittest.TestCase):

    def set_paddle_model(self):
        paddle_model = ViT_base_patch16_224()
        model_path = get_weights_path_from_url(
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams'
        )
        paddle_model.set_dict(paddle.load(model_path))
        return paddle_model

    def test_cv(self):
        paddle_model = self.set_paddle_model()

        img_path = 'tutorials/assets/catdog.png'
        algo = it.RolloutInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00429746, 0.00077522, 0.00285467, 0.00699459])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_layer(self):
        paddle_model = self.set_paddle_model()

        img_path = 'tutorials/assets/catdog.png'
        algo = it.RolloutInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, start_layer=1, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00487551, 0.00092406, 0.00239019, 0.00717532])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_multiple_inputs(self):
        paddle_model = self.set_paddle_model()

        img_path = ['tutorials/assets/catdog.png', 'tutorials/assets/catdog.png']
        algo = it.RolloutInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([4.29746090e-03, 7.75220629e-04, 2.85467086e-03, 6.99458644e-03,
        2.00000000e+00, 1.40000000e+01, 1.40000000e+01])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()

