import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestSG(unittest.TestCase):

    def test_cv(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.SmoothGradInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-2.4886960e-08,  1.4570465e-05, -2.3051113e-04,  2.0418176e-04])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_class(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.SmoothGradInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, labels=282, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-5.9705066e-08,  4.2906806e-05, -5.2132754e-04,  5.2274414e-04])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_multiple_inputs(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.SmoothGradInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-2.3143537e-08,  1.4913388e-05, -2.0887403e-04,  2.5168285e-04])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()