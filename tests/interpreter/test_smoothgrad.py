import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestSG(unittest.TestCase):

    def test_shape(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.SmoothGradInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, n_samples=1, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([1, 3, 224, 224]))    

    def test_visual(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.

        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.SmoothGradInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, n_samples=1, resize_to=256, crop_to=224, 
            visual=True, save_path='tmp.jpg')
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([1, 3, 224, 224]))    
        os.remove('tmp.jpg')

    def test_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.SmoothGradInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, n_samples=5, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 5.1930110e-06,  4.0496187e-03, -2.3638349e-02,  3.2775488e-02])

        assert_arrays_almost_equal(self, result, desired)

    def test_shape_v2(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.SmoothGradInterpreterV2(paddle_model, device='cpu')
        exp = algo.interpret(img_path, n_samples=2, resize_to=256, crop_to=224, visual=False, save_path='tmp.jpg')
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([1, 3, 224, 224]))
        os.remove('tmp.jpg')

    def test_shape_split_v2(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.SmoothGradInterpreterV2(paddle_model, device='cpu')
        exp = algo.interpret(img_path, n_samples=2, split=1, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([1, 3, 224, 224]))

if __name__ == '__main__':
    unittest.main()