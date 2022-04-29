import paddle
import unittest
import numpy as np
import interpretdl as it

from tests.utils import assert_arrays_almost_equal

from interpretdl.common.file_utils import download_and_decompress
import subprocess
import sys


class TestGA(unittest.TestCase):
    def set_paddle_model(self):
        url = 'https://github.com/PaddlePaddle/InterpretDL/files/8589193/clip.zip'
        download_and_decompress(url, './')
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./clip/requirements.txt"])
        from clip import tokenize, load_model

        paddle_model, transforms = load_model('ViT_B_32', pretrained=True)
        return paddle_model, tokenize

    def test_algo(self):
        paddle_model, tokenize = self.set_paddle_model()
        img_path = 'tutorials/assets/catdog.png'
        texts = ["a cat"]
        text_tokenized = tokenize(texts)
        algo = it.GAInterpreter(paddle_model, device='cpu')
        R_txt, R_img = algo.interpret(img_path, texts, text_tokenized, crop_to=224, visual=False)

        result = np.array([R_img.mean(), R_img.std(), R_img.min(), R_img.max()])
        desired = np.array([0.00365583, 0.00676875, 0.00013451, 0.03150804], dtype=np.float32)
        assert_arrays_almost_equal(self, result, desired)        


if __name__ == '__main__':
    unittest.main()

