import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestLIME(unittest.TestCase):

    def test_shape_and_algo(self):

        from interpretdl.common.file_utils import download_and_decompress
        download_and_decompress("https://github.com/PaddlePaddle/InterpretDL/files/8837286/glime_avg_normlime_global_weights.zip")
        
        paddle_model = mobilenet_v2(pretrained=True)
        algo = it.GLIMECVInterpreter(paddle_model)
        algo.set_global_weights(algo)
        algo.set_global_weights("imagenet_global_weights_avg.npy")

        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        exp = algo.interpret(img_path, prior_method='ridge', num_samples=50, batch_size=50, resize_to=64, crop_to=64, visual=False)

        algo.compute_global_weights('normlime', [algo.lime_results])

        k = list(exp.keys())[0]
        result = np.zeros(len(exp[k]))
        for sp_id, v in exp[k]:
            result[sp_id] = v        
        
        result = np.array([k, *result])

        desired = np.array([4.90000000e+02,  5.57969743e-02,  1.55487054e-02,  3.55165317e-02,
        1.40118726e-02,  6.44251208e-02,  3.98409511e-03, -4.13299968e-03,
       -3.18464635e-02, -2.44578178e-02,  1.69825564e-02,  1.46489598e-03,
        5.90007685e-02,  2.48074096e-02, -7.30806851e-02, -4.49090874e-02,
        1.59183040e-02, -2.56270650e-02,  9.35910465e-02,  3.60380997e-02,
        1.34488334e-01,  6.50497344e-02,  1.77734308e-02,  3.19729481e-02,
        4.52839121e-02,  9.33561008e-02,  1.15080587e-02,  1.38845717e-02,
        9.43465624e-02,  2.92089127e-02,  3.20564642e-02, -1.74206462e-03,
        8.34846462e-02,  2.95507758e-02,  3.25349795e-02, -1.97837795e-03,
        1.27033323e-01, -9.81423012e-03,  3.89915593e-02,  2.96282821e-02,
        7.63927405e-02,  1.74757999e-02,  9.21592761e-02,  3.58518795e-02,
        3.37004215e-02,  6.16672433e-02,  7.55628559e-02,  5.53482498e-02,
        7.44542520e-03,  1.64932902e-02,  1.10531985e-02,  8.53582658e-02,
        6.09448481e-02, -2.51819750e-03,  7.52857977e-02, -2.16946058e-02,
       -3.11996488e-02, -7.93963945e-02,  6.08324498e-02,  4.14586328e-02,
        1.84622894e-02,  2.46146574e-02, -1.73424450e-02, -7.64635222e-03])

        assert_arrays_almost_equal(self, result, desired)

        os.remove('glime_avg_normlime_global_weights.zip')
        os.remove('imagenet_global_weights_avg.npy')
        os.remove('imagenet_global_weights_normlime.npy')


if __name__ == '__main__':
    unittest.main()