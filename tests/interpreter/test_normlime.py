import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import glob
import os
import paddle
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestNormLIME(unittest.TestCase):

    def test_shape_and_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)
        normlime = it.NormLIMECVInterpreter(paddle_model, 'cpu')
        dataset_dir = "tutorials/assets"
        image_paths = sorted(glob.glob(dataset_dir + "/*.png"))
        normlime.interpret(image_paths, num_samples=20, batch_size=20)
        paddle.disable_static()

        os.remove('all_lime_weights.npz')
        os.remove('normlime_weights.npy')

    def test_prior_shape_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)
        algo = it.LIMEPriorInterpreter(paddle_model, prior_method='ridge', device='cpu')
        dataset_dir = "tutorials/assets"
        image_paths = sorted(glob.glob(dataset_dir + "/*.png"))
        algo.interpreter_init(image_paths, batch_size=20)
        
        algo.interpret(image_paths[0], num_samples=20, batch_size=20, resize_to=64, crop_to=64, visual=False)


if __name__ == '__main__':
    unittest.main()