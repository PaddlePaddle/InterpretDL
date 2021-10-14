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
        desired = np.array([-1.16835075e-07,  8.02171271e-05, -1.70097279e-03,  1.08779408e-03])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_class(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradShapCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, labels=282, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-2.0223466e-08,  2.0590464e-04, -7.1432171e-03,  4.5942976e-03])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_multiple_inputs(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.GradShapCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-6.147997e-09,  5.921054e-05, -1.255356e-03,  1.409926e-03])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()