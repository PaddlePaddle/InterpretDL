import unittest
import numpy as np
from numpy.core.fromnumeric import size
import paddle
import interpretdl as it
from interpretdl.common.file_utils import *
from interpretdl.common.paddle_utils import FeatureExtractor
from tests.utils import assert_arrays_almost_equal


class TestFileUtils(unittest.TestCase):
    def test_file_utils(self):
        download_and_decompress("https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz")
        assert md5check('pre_models.tar.gz', '9375cab3b7200365b01b1dd2bc025935')
        assert md5check('pre_models.tar.gz', 'aaa') is False
        os.remove('pre_models.tar.gz')
    
    def test_mv(self):
        os.makedirs('tmp/s1')
        os.makedirs('tmp/s2')
        move_and_merge_tree('tmp', 'tmp')
        shutil.rmtree('tmp')

class TestPaddleUtils(unittest.TestCase):
    def test_feature_extractor(self):
        fe = FeatureExtractor()
        paddle.enable_static()
        fe.session_prepare()
        paddle.disable_static()
        shutil.rmtree('pre_models')


if __name__ == '__main__':
    unittest.main()