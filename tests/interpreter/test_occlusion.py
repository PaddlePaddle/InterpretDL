import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestOCC(unittest.TestCase):

    def test_cv(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.OcclusionInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(
            img_path, sliding_window_shapes=(1, 20, 20), strides=(1, 20, 20), visual=False
        )
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 1.19814882e-02,  5.36160879e-02, -1.64756835e-01,  1.99667364e-01,
        1.00000000e+00,  3.00000000e+00,  2.24000000e+02,  2.24000000e+02])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_class(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.OcclusionInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(
            img_path, labels=282, sliding_window_shapes=(1, 20, 20), strides=(1, 20, 20), visual=False
        )
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 9.92860179e-03,  4.55657057e-02, -1.00801319e-01,  2.11558089e-01,
        1.00000000e+00,  3.00000000e+00,  2.24000000e+02,  2.24000000e+02])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_multiple_inputs(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.OcclusionInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, sliding_window_shapes=(1, 20, 20), strides=(1, 20, 20), visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 1.19814882e-02,  5.36160879e-02, -1.64756835e-01,  1.99667364e-01,
            2.00000000e+00,  3.00000000e+00,  2.24000000e+02,  2.24000000e+02])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()