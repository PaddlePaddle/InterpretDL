import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestLIME(unittest.TestCase):

    def test_shape_and_algo(self):
        paddle_model = mobilenet_v2(pretrained=True)
        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.LIMECVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, num_samples=100, batch_size=50, resize_to=64, crop_to=64, visual=False)
        k = list(exp.keys())[0]
        result = np.zeros(len(exp[k]))
        for sp_id, v in exp[k]:
            result[sp_id] = v        
        
        result = np.array([k, *result])
        desired = np.array([ 4.90000000e+02,  1.86663490e-03, -6.07835046e-02,  3.15221286e-02,
                -4.33815891e-02,  5.98708568e-02, -4.03566565e-02,  2.32181935e-04,
                -1.88198336e-02, -5.48955918e-04, -3.05450687e-02,  6.40058578e-02,
                6.46084885e-02, -1.35210916e-03, -3.33294607e-02, -3.67286859e-02,
                3.36535163e-02, -8.59251274e-03,  4.58715622e-02,  4.07490782e-02,
                1.16548644e-01,  8.28152785e-02,  6.30378504e-02, -9.37234923e-03,
                5.44956613e-03,  2.78750946e-03, -5.23468125e-02,  2.56323242e-03,
                9.42740024e-02,  7.96398789e-02,  6.01124075e-02, -5.76994244e-02,
                7.03640418e-02,  3.61672162e-02,  4.97282116e-02,  5.11243389e-02,
                1.89338674e-01,  1.21587496e-02,  4.50969712e-02,  4.63132584e-02,
                -4.05242203e-03,  5.21016618e-02,  7.57953552e-02,  2.91536810e-02,
                1.51803549e-02,  6.08129130e-02,  5.33129929e-02, -2.09711908e-02,
                -4.93060550e-03, -1.41900513e-02,  7.03506893e-02,  5.65046961e-02,
                -2.94891549e-02, -3.20462166e-02,  2.93698146e-02, -1.15419265e-01,
                -8.70454571e-03, -3.36799397e-02,  2.18574095e-02,  1.36021779e-02,
                4.22237924e-02, -7.91892625e-02,  1.59767297e-02,  2.24134222e-02])

        assert_arrays_almost_equal(self, result, desired)

    def test_save(self):
        import matplotlib
        matplotlib.use('agg')  # non-GUI, for skipping.

        paddle_model = mobilenet_v2(pretrained=True)
        np.random.seed(42)
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        algo = it.LIMECVInterpreter(paddle_model, device='cpu')
        exp = algo.interpret(img_path, num_samples=42, batch_size=20, 
            resize_to=64, crop_to=64, visual=True, save_path='tmp.jpg')
        os.remove('tmp.jpg')


if __name__ == '__main__':
    unittest.main()