import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
from paddle.vision.models.resnet import resnet50

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestGradCAM(unittest.TestCase):

    def test_cv(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.18.2', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([7.08578909e-06, 9.28105146e-06, 0.00000000e+00, 3.74892770e-05,
            1.00000000e+00, 7.00000000e+00, 7.00000000e+00])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_class(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.18.2', label=282, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([5.12873930e-06, 7.74075761e-06, 0.00000000e+00, 2.88265182e-05,
        1.00000000e+00, 7.00000000e+00, 7.00000000e+00])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_layer(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.16.conv.3', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([2.97199367e-05, 3.79896701e-05, 0.00000000e+00, 1.25247447e-04,
            1.00000000e+00, 7.00000000e+00, 7.00000000e+00])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_layer_2(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.GradCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.8.conv.3', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([1.13254619e-05, 1.62324668e-05, 0.00000000e+00, 6.76311683e-05,
            1.00000000e+00, 1.40000000e+01, 1.40000000e+01])

        assert_arrays_almost_equal(self, result, desired, 2e-3)

    def test_cv_multiple_inputs(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.GradCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.18.2', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([7.08578864e-06, 9.28105146e-06, 0.00000000e+00, 3.74892770e-05,
            2.00000000e+00, 7.00000000e+00, 7.00000000e+00])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()