import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestIG(unittest.TestCase):

    def test_shape(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        algo = it.IntGradCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=3, num_random_trials=2, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([1, 3, 224, 224]))    

    def test_cv_multiple_inputs(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = ['imgs/catdog.jpg', 'imgs/catdog.jpg']
        algo = it.IntGradCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=3, num_random_trials=2, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])

        assert_arrays_almost_equal(self, result, np.array([2, 3, 224, 224]))

    def test_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.IntGradCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=3, num_random_trials=2, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        print(repr(result))
        desired = np.array([ 4.1104166e-05,  2.1593589e-03, -1.8412363e-02,  2.2863150e-02])

        assert_arrays_almost_equal(self, result, desired)

    def test_algo_random(self):
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.IntGradCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=3, num_random_trials=2, baselines='random', visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 4.9712045e-05,  2.4339566e-03, -1.9965716e-02,  2.3791827e-02])

        assert_arrays_almost_equal(self, result, desired)

    def test_algo_2(self):
        paddle_model = mobilenet_v2(pretrained=True)
        np.random.seed(42)
        algo = it.IntGradNLPInterpreter(paddle_model, device='cpu')
        data = np.random.random(size=(1, 3, 64, 64)).astype(np.float32)
        exp = algo.interpret(data, steps=3, embedding_name='features.18.2', return_pred=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([ 4.4516042e-07,  4.2583229e-06, -2.4716886e-05,  8.8872534e-05])
        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.
        paddle_model = mobilenet_v2(pretrained=True)

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.IntGradCVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=3, num_random_trials=2, visual=True, save_path='tmp.jpg')
        os.remove('tmp.jpg')

if __name__ == '__main__':
    unittest.main()