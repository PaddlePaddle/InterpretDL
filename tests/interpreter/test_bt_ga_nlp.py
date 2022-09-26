import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestGABTNLP(unittest.TestCase):

    def prepare(self):
        import paddle
        import paddlenlp
        from paddlenlp.data import Stack, Tuple, Pad
        from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
        from tutorials.assets.utils import convert_example, aggregate_subwords_and_importances, layer_replacement
        
        MODEL_NAME = "ernie-2.0-base-en"
        paddle.device.set_device('cpu')
        model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)
        tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)
        model = layer_replacement(model)

        self.paddle_model = model
        self.tokenizer = tokenizer

    def test_bt_ga_nlp(self):
        self.prepare()

        reviews = [
            "it 's a charming and often affecting journey . ",
        ]

        algo = it.BTNLPInterpreter(self.paddle_model, device='cpu')
        for i, review in enumerate(reviews):
            subword_importances = algo.interpret(
                review,
                tokenizer=self.tokenizer,
                ap_mode="token",
                label=1,
                start_layer=9,
                steps=2)
            
            subword_importances = algo.interpret(
                review,
                tokenizer=self.tokenizer,
                ap_mode="head",
                label=1,
                start_layer=11,
                steps=2)
            
        algo = it.GANLPInterpreter(self.paddle_model, device='cpu')
        for i, review in enumerate(reviews):
            subword_importances = algo.interpret(
                review,
                tokenizer=self.tokenizer,
                label=1,
                start_layer=11)

if __name__ == '__main__':
    unittest.main()
