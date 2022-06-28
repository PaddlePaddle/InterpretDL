import paddle
import unittest
import numpy as np
import interpretdl as it
from paddle.utils.download import get_weights_path_from_url

from tutorials.assets.vision_transformer import ViT_base_patch16_224
from tests.utils import assert_arrays_almost_equal


class TestBT(unittest.TestCase):

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
        algo = it.BTInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=2, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])
        assert_arrays_almost_equal(self, result, np.array([1, 14, 14]))

    def test_cv_multiple_inputs(self):
        paddle_model = self.set_paddle_model()

        img_path = ['tutorials/assets/catdog.png', 'tutorials/assets/catdog.png']
        algo = it.BTInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=2, resize_to=256, crop_to=224, visual=False)
        result = np.array([*exp.shape])
        assert_arrays_almost_equal(self, result, np.array([2, 14, 14]))

    def test_algo(self):
        paddle_model = self.set_paddle_model()

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
        algo = it.BTInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, steps=2, visual=False)
        result = np.array([exp.mean(), exp.std(), exp.min(), exp.max()])
        desired = np.array([0.00511069, 0.00290341, 0.00061763, 0.01646499], dtype=float32)

        assert_arrays_almost_equal(self, result, desired)        


if __name__ == '__main__':
    unittest.main()
