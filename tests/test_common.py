import unittest
import numpy as np
from numpy.core.fromnumeric import size

import interpretdl as it
from interpretdl.common.file_utils import *
from interpretdl.common.paddle_utils import FeatureExtractor
from tests.utils import assert_arrays_almost_equal


class TestFileUtils(unittest.TestCase):
    def test_file_utils(self):
        download_and_decompress("https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz")
        assert md5check('pre_models.tar.gz', '9375cab3b7200365b01b1dd2bc025935')
        os.remove('pre_models.tar.gz')


class TestPaddleUtils(unittest.TestCase):
    def test_feature_extractor(self):
        fe = FeatureExtractor()
        fe.session_prepare()


if __name__ == '__main__':
    unittest.main()