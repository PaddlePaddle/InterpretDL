import unittest
from paddle.vision.models import mobilenet_v2, resnet50
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestScoreCAM(unittest.TestCase):
    
    def test_cv(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.ScoreCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.18.2', visual=False)

        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 67.57363816,  39.71009002,   3.64695652, 144.3522944 ,
         1.        , 224.        , 224.        ])

        assert_arrays_almost_equal(self, result, desired)
    
    def test_cv_class(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.ScoreCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.18.2', labels=282, visual=False)

        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 48.47917838,  27.82020997,   3.85096038, 116.12830776,
         1.        , 224.        , 224.        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_layer(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.ScoreCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.16.conv.3', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([  0.45457669,   1.73647004,  -3.54019404,   3.77973473,
         1.        , 224.        , 224.        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_layer_2(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.ScoreCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.8.conv.3', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 1.04185688e-01,  1.23888987e+00, -4.07477156e+00,  2.84238683e+00,
        1.00000000e+00,  2.24000000e+02,  2.24000000e+02])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_multiple_inputs(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.ScoreCAMInterpreter(paddle_model, use_cuda=False)
        exp = algo.interpret(img_path, 'features.18.2', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max(), *exp.shape])
        desired = np.array([ 67.57363816,  39.71009002,   3.64695652, 144.3522944 ,
         2.        , 224.        , 224.        ])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()