import unittest
import numpy as np

import interpretdl as it
from paddle.vision.models import mobilenet_v3_small

from tests.utils import assert_arrays_almost_equal


class TestLoc(unittest.TestCase):

    def test_evaluate(self):
        paddle_model = mobilenet_v3_small(pretrained=True)
        img_path = 'imgs/catdog.jpg'
        gradcam = it.GradCAMInterpreter(paddle_model, device='cpu')
        exp = gradcam.interpret(
            img_path,
            'lastconv.2',
            visual=False, 
            save_path=None)
        evaluator = it.PointGame()
        r = evaluator.evaluate([0, 0, 3, 3], exp)

        desired = list({'precision': 1.0,
                'recall': 0.19047619047619047,
                'f1_score': 0.3199997312002258,
                'auc_score': 0.6054421768707483,
                'ap_score': 0.6184073259558627}.values())

        assert_arrays_almost_equal(self, np.array(list(r.values())), np.array(desired))

        evaluator = it.PointGameSegmentation()
        gt_seg = np.zeros_like(exp)
        gt_seg[0, :3, 2:5] = 1
        r = evaluator.evaluate(gt_seg, exp)

        desired = list({'precision': 1.0,
            'recall': 0.44444445,
            'f1_score': 0.615384192523618,
            'auc_score': 0.6416666666666667,
            'ap_score': 0.608843537414966}.values())

        assert_arrays_almost_equal(self, np.array(list(r.values())), np.array(desired))


if __name__ == '__main__':
    unittest.main()