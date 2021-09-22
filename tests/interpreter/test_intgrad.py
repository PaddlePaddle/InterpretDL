import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestIG(unittest.TestCase):

    def test_cv(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.IntGradCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 3.0684089e-06,  1.9912077e-03, -3.8767897e-02,  4.7322020e-02])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_class(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.IntGradCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, labels=282, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 2.5876125e-06,  1.4400641e-03, -2.1012696e-02,  2.6807360e-02])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_multiple_inputs(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.IntGradCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 3.0684084e-06,  1.9912077e-03, -3.8767897e-02,  4.7322020e-02])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()