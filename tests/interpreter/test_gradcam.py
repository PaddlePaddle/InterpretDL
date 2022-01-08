import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
from paddle.vision.models.resnet import resnet50
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestGradCAM(unittest.TestCase):

    def test_shape(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.18.2', resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])
        
        assert_arrays_almost_equal(self, result, np.array([1, 7, 7]))

    def test_cv_multiple_inputs(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.18.2', resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])
        desired = np.array([2, 7, 7])

        assert_arrays_almost_equal(self, result, desired)

    def test_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.18.2', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([2.4055550e-04, 9.1252205e-06, 2.2793890e-04, 2.5061620e-04], 
                dtype=np.float32)

        assert_arrays_almost_equal(self, result, desired)

    def test_algo_layer(self):
        paddle_model = mobilenet_v2(pretrained=True)
        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.16.conv.3', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([1.07594451e-03, 4.04298247e-04, 5.23229188e-04, 1.60590897e-03])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, 'features.18.2', visual=True, save_path='tmp.jpg')
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([2.4055550e-04, 9.1252205e-06, 2.2793890e-04, 2.5061620e-04], 
                dtype=np.float32)

        assert_arrays_almost_equal(self, result, desired)

        os.remove('tmp.jpg')

if __name__ == '__main__':
    unittest.main()