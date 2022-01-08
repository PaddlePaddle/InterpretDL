import unittest
from paddle.vision.models import mobilenet_v2, resnet50
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestScoreCAM(unittest.TestCase):
    
    def test_shape(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.ScoreCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.18.2', resize_to=64, crop_to=64, visual=False)

        result = np.array([*exp.shape])
        desired = np.array([1.        , 64.        , 64.        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.ScoreCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.16.conv.3', resize_to=64, crop_to=64, visual=False)

        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 2.3779001 ,  1.80234987, -0.33591489,  5.09665543])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.

        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.ScoreCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.16.conv.3', resize_to=64, crop_to=64, 
            visual=True, save_path='tmp.jpg')

        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 2.3779001 ,  1.80234987, -0.33591489,  5.09665543])

        assert_arrays_almost_equal(self, result, desired)
        os.remove('tmp.jpg')

if __name__ == '__main__':
    unittest.main()