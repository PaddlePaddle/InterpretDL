import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestLIME(unittest.TestCase):

    def test_cv(self):
        paddle_model = mobilenet_v2(pretrained=True)

        img_path = 'imgs/catdog.jpg'
        algo = it.LIMECVInterpreter(paddle_model, model_input_shape=[3, 64, 64], use_cuda=False, random_seed=42)
        exp = algo.interpret(img_path, num_samples=200, batch_size=50, visual=False)
        result = np.zeros(len(exp[537]))
        for sp_id, v in exp[537]:
            result[sp_id] = v
        desired = np.array([ 0.12754734,  0.05717407,  0.11495698,  0.05202468,  0.12741461,
            -0.03031597,  0.05229854,  0.08469778,  0.06987132])

        assert_arrays_almost_equal(self, result, desired)

    def test_cv_class(self):
        paddle_model = mobilenet_v2(pretrained=True)
        interpret_class = 282

        img_path = 'imgs/catdog.jpg'
        algo = it.LIMECVInterpreter(paddle_model, model_input_shape=[3, 64, 64], use_cuda=False, random_seed=42)
        exp = algo.interpret(img_path, interpret_class=interpret_class, num_samples=200, batch_size=50, visual=False)
        result = np.zeros(len(exp[interpret_class]))
        for sp_id, v in exp[interpret_class]:
            result[sp_id] = v
        desired = np.array([ 6.93271851e-04,  4.66737058e-04,  1.01167932e-03, -7.85369867e-04,
            -1.05476870e-05,  2.14875737e-05, -9.73834840e-04, -6.43646977e-04,
            -1.92199272e-04])

        assert_arrays_almost_equal(self, result, desired)

if __name__ == '__main__':
    unittest.main()