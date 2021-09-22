import unittest
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal
from tutorials.assets.lrp_model import resnet50

class TestLRP(unittest.TestCase):

    def test_cv(self):
        paddle_model = resnet50(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.LRPCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([1.8663127e-04, 2.1888215e-04, 1.7363816e-06, 3.2938947e-03])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_class(self):
        paddle_model = resnet50(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.LRPCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, label=282, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([1.5382419e-04, 1.4931166e-04, 2.2044858e-06, 2.2588256e-03])

        assert_arrays_almost_equal(self, result, desired)


    def test_cv_multiple_inputs(self):
        paddle_model = resnet50(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.LRPCVInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([1.8663125e-04, 2.1888215e-04, 1.7363816e-06, 3.2938947e-03])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()
