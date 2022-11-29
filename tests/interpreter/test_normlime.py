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

        # compute but not save intermediate results.
        normlime.interpret(image_paths, num_samples=20, batch_size=20, temp_data_file=None)

        # compute and save
        normlime.interpret(image_paths, num_samples=20, batch_size=20)

        # load already computed one.
        normlime.interpret(image_paths, num_samples=20, batch_size=20)

        os.remove('all_lime_weights.npz')
        os.remove('normlime_weights.npy')
        os.remove('normlime_weights-0.npy')
        os.remove('normlime_weights-1.npy')


if __name__ == '__main__':
    unittest.main()