import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestGradiantSHAP(unittest.TestCase):

    def test_cv(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradShapCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-5.8417476e-07,  4.0108559e-04, -8.5048620e-03,  5.4389671e-03])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_class(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradShapCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, labels=282, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-1.0111845e-07,  1.0295232e-03, -3.5716087e-02,  2.2971490e-02])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_multiple_inputs(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.GradShapCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-6.1479746e-08,  5.9210538e-04, -1.2553556e-02,  1.4099267e-02])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()