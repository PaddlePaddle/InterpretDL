import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestGradiantSHAP(unittest.TestCase):

    def test_shape(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.GradShapCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([1, 3, 224, 224]))    

    def test_cv_multiple_inputs(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.GradShapCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, resize_to=256, crop_to=224, visual=False, n_samples=1)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([2, 3, 224, 224]))

    def test_algo(self):
        np.random.seed(42)
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.GradShapCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 3.4148130e-07,  2.6306865e-05, -2.3330076e-04,  3.4065323e-04])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.

        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.GradShapCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, visual=True, save_path='tmp.jpg')
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 3.4148130e-07,  2.6306865e-05, -2.3330076e-04,  3.4065323e-04])

        assert_arrays_almost_equal(self, result, desired)
        os.remove('tmp.jpg')


if __name__ == '__main__':
    unittest.main()