import paddle
import unittest
import numpy as np
import interpretdl as it
from paddle.utils.download import get_weights_path_from_url
import os
from tutorials.assets.vision_transformer import ViT_base_patch16_224
from tests.utils import assert_arrays_almost_equal


class TestRollout(unittest.TestCase):

    def set_paddle_model(self):
        paddle_model = ViT_base_patch16_224()
        model_path = get_weights_path_from_url(
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams'
        )
        paddle_model.set_dict(paddle.load(model_path))
        return paddle_model

    def test_shape(self):
        paddle_model = self.set_paddle_model()
        img_path = 'tutorials/assets/catdog.png'
        algo = it.RolloutInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])
        assert_arrays_almost_equal(self, result, np.array([1, 14, 14]))

    def test_cv_multiple_inputs(self):
        paddle_model = self.set_paddle_model()

        img_path = ['tutorials/assets/catdog.png', 'tutorials/assets/catdog.png']
        algo = it.RolloutInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])
        assert_arrays_almost_equal(self, result, np.array([2, 14, 14]))

    def test_algo(self):
        paddle_model = self.set_paddle_model()

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
        algo = it.RolloutInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00437814, 0.00064302, 0.00304579, 0.00634555])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.
        paddle_model = self.set_paddle_model()

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
        algo = it.RolloutInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, visual=True, save_path='tmp.jpg')
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00437814, 0.00064302, 0.00304579, 0.00634555])

        assert_arrays_almost_equal(self, result, desired)
        os.remove('tmp.jpg')

if __name__ == '__main__':
    unittest.main()

