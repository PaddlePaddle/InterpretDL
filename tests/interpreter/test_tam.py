import paddle
import unittest
import numpy as np
import interpretdl as it
from paddle.utils.download import get_weights_path_from_url

from tutorials.assets.vision_transformer import ViT_base_patch16_224
from tests.utils import assert_arrays_almost_equal


class TestTAM(unittest.TestCase):

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
        algo = it.TAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.01317265, 0.01318072, 0.00068325, 0.06299927])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_layer(self):
        paddle_model = self.set_paddle_model()

        img_path = 'tutorials/assets/catdog.png'
        algo = it.TAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, start_layer=1, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.10222234, 0.0989774, 0.00654205, 0.45563712])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_class(self):
        paddle_model = self.set_paddle_model()

        img_path = 'tutorials/assets/catdog.png'
        algo = it.TAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, label=243, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([8.54372094e-03, 1.24013127e-02, 9.54663476e-05, 6.53232618e-02])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_steps(self):
        paddle_model = self.set_paddle_model()

        img_path = 'tutorials/assets/catdog.png'
        algo = it.TAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, steps=10, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.01300348, 0.01331104, 0.00072738, 0.06281879])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_multiple_inputs(self):
        paddle_model = self.set_paddle_model()

        img_path = ['tutorials/assets/catdog.png', 'tutorials/assets/catdog.png']
        algo = it.TAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([1.31726510e-02, 1.31807191e-02, 6.83252022e-04, 6.29992668e-02, 2.00000000e+00, 1.40000000e+01, 1.40000000e+01])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()

