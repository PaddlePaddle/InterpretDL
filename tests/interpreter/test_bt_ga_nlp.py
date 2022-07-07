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

        def preprocess_fn(data):
            examples = []
            
            if not isinstance(data, list):
                data = [data]
            
            for text in data:
                input_ids, segment_ids = convert_example(
                    text,
                    tokenizer,
                    max_seq_length=128,
                    is_test=True
                )
                examples.append((input_ids, segment_ids))

            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
                Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
            ): fn(samples)
            
            input_ids, segment_ids = batchify_fn(examples)
            return paddle.to_tensor(input_ids, stop_gradient=False), paddle.to_tensor(segment_ids, stop_gradient=False)
        self.preprocess_fn = preprocess_fn

    def test_bt_ga_nlp(self):
        self.prepare()

        reviews = [
            "it 's a charming and often affecting journey . ",
        ]
        data = [ {"text": r} for r in reviews]

        algo = it.BTNLPInterpreter(self.paddle_model, device='cpu')
        for i, review in enumerate(data):
            subword_importances = algo.interpret(
                ap_mode="token",
                data=self.preprocess_fn(data),
                label=1,
                start_layer=9)
            
            subword_importances = algo.interpret(
                data=self.preprocess_fn(data),
                label=1,
                start_layer=11)
            
        algo = it.GANLPInterpreter(self.paddle_model, device='cpu')
        for i, review in enumerate(data):
            subword_importances = algo.interpret(
                data=self.preprocess_fn(data),
                label=1,
                start_layer=11)


if __name__ == '__main__':
    unittest.main()
