import unittest
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal
from tutorials.assets.lrp_model import resnet50

class TestLRP(unittest.TestCase):

    def test_shape(self):
        paddle_model = resnet50(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.LRPCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, resize_to=64, crop_to=64, visual=False)
        result = np.array([*exp.shape])
        
        assert_arrays_almost_equal(self, result, np.array([1, 1, 64, 64]))

    def test_cv_multiple_inputs(self):
        paddle_model = resnet50(pretrained=True)

        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.LRPCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, resize_to=64, crop_to=64, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([2, 1, 64, 64]))

    def test_algo(self):
        paddle_model = resnet50(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.LRPCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00397945, 0.00368139, 0.00014466, 0.03765493])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.

        paddle_model = resnet50(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.LRPCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, visual=True, save_path='tmp.jpg')
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00397945, 0.00368139, 0.00014466, 0.03765493])

        assert_arrays_almost_equal(self, result, desired)
        os.remove('tmp.jpg')

if __name__ == '__main__':
    unittest.main()
