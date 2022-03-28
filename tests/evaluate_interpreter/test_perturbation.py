import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
from paddle.vision.models.resnet import resnet50

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestDelIns(unittest.TestCase):

    def test_evaluate_sg(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        sg = it.SmoothGradInterpreter(paddle_model, device='cpu')
        exp = sg.interpret(
            img_path,
            n_samples=5,
            noise_amount=0.1,
            visual=False, 
            labels=None, 
            save_path=None)
        evaluator = it.Perturbation(paddle_model, device='cpu')
        r = evaluator.evaluate(img_path, exp)
        # test ends if no errors

    def test_evaluate_lime(self):
        paddle_model = mobilenet_v2(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        lime = it.LIMECVInterpreter(paddle_model, device='cpu')
        lime_weights = lime.interpret(
            img_path,
            num_samples=50,
            batch_size=50,
            visual=False,
            save_path=None
        )
        evaluator = it.Perturbation(paddle_model, device='cpu')
        r = evaluator.evaluate(img_path, lime.lime_results)
        # test ends if no errors

if __name__ == '__main__':
    unittest.main()