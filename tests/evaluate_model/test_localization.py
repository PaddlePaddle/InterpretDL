import unittest
import numpy as np

import interpretdl as it
from paddle.vision.models import mobilenet_v3_small

from tests.utils import assert_arrays_almost_equal


class TestLoc(unittest.TestCase):

    def test_evaluate(self):
        paddle_model = mobilenet_v3_small(pretrained=True)
        np.random.seed(42)
        # jpeg decoding may be slightly different because of version and device.
        img_path = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
        gradcam = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = gradcam.interpret(
            img_path,
            'lastconv.2',
            visual=False, 
            save_path=None)
        evaluator = it.PointGame()
        r = evaluator.evaluate([0, 0, 6, 6], exp, threshold=0.01)

        desired = list({'precision': 0.8571428571428571, 
            'recall': 1.0, 
            'f1_score': 0.9230764260357706, 
            'auc_score': 0.6122448979591837, 
            'ap_score': 0.9242890900837504}.values())

        assert_arrays_almost_equal(self, np.array(list(r.values())), np.array(desired))

        evaluator = it.PointGameSegmentation()
        gt_seg = np.zeros_like(exp)
        gt_seg[0, :3, 2:5] = 1
        r = evaluator.evaluate(gt_seg, exp)

        desired = list({'precision': 0.19148936170212766, 
            'recall': 1.0, 
            'f1_score': 0.3214283016583897, 
            'auc_score': 0.5611111111111111, 
            'ap_score': 0.2282429182374856}.values())

        assert_arrays_almost_equal(self, np.array(list(r.values())), np.array(desired))


if __name__ == '__main__':
    unittest.main()