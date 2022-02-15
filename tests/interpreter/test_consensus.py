import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestConsensus(unittest.TestCase):

    def test_algo(self):
        from paddle.vision.models import resnet34, resnet50, resnet101, mobilenet_v2

        # Here we use four models to give an illustration. Using more models shows more impressive results.
        list_models = {
            'resnet34': resnet34(pretrained=False), 
            'resnet50': resnet50(pretrained=False),
            'resnet101': resnet101(pretrained=False), 
            'mobilenet_v2': mobilenet_v2(pretrained=False)
        }
        consensus = it.ConsensusInterpreter(it.SmoothGradInterpreter, list_models.values(), device='gpu:0')
        img_path = np.random.randint(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        exp = consensus.interpret(img_path, n_samples=5)


if __name__ == '__main__':
    unittest.main()