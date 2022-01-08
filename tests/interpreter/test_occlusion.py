import unittest
from paddle.vision.models import resnet50
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestOCC(unittest.TestCase):

    def test_shape(self):
        paddle_model = resnet50(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.OcclusionInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(
            img_path, sliding_window_shapes=(1, 32, 32), strides=(1, 32, 32), 
            resize_to=64, crop_to=64, visual=False)
        result = np.array([*exp.shape])
        assert_arrays_almost_equal(self, result, np.array([1, 3, 64, 64]))

    def test_cv_multiple_inputs(self):
        paddle_model = resnet50(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.OcclusionInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(
            img_path, sliding_window_shapes=(1, 32, 32), strides=(1, 32, 32), 
            resize_to=64, crop_to=64, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([2, 3, 64, 64]))

    def test_algo(self):
        paddle_model = resnet50(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.OcclusionInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(
            img_path, sliding_window_shapes=(1, 32, 32), strides=(1, 32, 32), 
            resize_to=64, crop_to=64, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-0.11518122,  0.16784915, -0.46479446,  0.11298946])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.

        paddle_model = resnet50(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.OcclusionInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(
            img_path, sliding_window_shapes=(1, 33, 33), strides=(1, 32, 32), 
            resize_to=64, crop_to=64, visual=True, save_path='tmp.jpg')
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([-0.12583457,  0.20028804, -0.8618185 ,  0.2236681 ])

        assert_arrays_almost_equal(self, result, desired)
        os.remove('tmp.jpg')


if __name__ == '__main__':
    unittest.main()