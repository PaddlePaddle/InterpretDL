import unittest
from paddle.vision.models import mobilenet_v2
import numpy as np
import os
import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestLIMENLP(unittest.TestCase):

    def prepare(self):
        import paddle
        import paddlenlp
        from paddlenlp.data import Stack, Tuple, Pad
        from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
        
        MODEL_NAME = "ernie-2.0-base-en"
        model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)
        tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

        self.paddle_model = model
        self.tokenizer = tokenizer

        def text_to_input_fn(raw_text):
            encoded_inputs = tokenizer(text=raw_text, max_seq_len=128)
            # order is important. *_batched_and_to_tuple will be the input for the model.
            _batched_and_to_tuple = tuple([np.array([v]) for v in encoded_inputs.values()])
            return _batched_and_to_tuple
        self.text_to_input_fn = text_to_input_fn

    def test_lime_intgrad_gradshap_nlp(self):
        self.prepare()

        reviews = [
            "it 's a charming and often affecting journey . ",
        ]

        for i, review in enumerate(reviews):
            algo = it.LIMENLPInterpreter(self.paddle_model, device='cpu')
            lime_weights = algo.interpret(
                review,
                tokenizer=self.tokenizer,
                num_samples=11,
                batch_size=10)

            algo = it.IntGradNLPInterpreter(self.paddle_model, device='cpu')
            avg_gradients = algo.interpret(
                review,
                tokenizer=self.tokenizer,
                steps=2)

            algo = it.SmoothGradNLPInterpreter(self.paddle_model, device='cpu')
            avg_gradients = algo.interpret(
                review,
                tokenizer=self.tokenizer,
                n_samples=2,
                noise_amount=0.1)
            
    def test_normlime(self):
        self.prepare()

        reviews = [
            "it 's a charming and often affecting journey . ",
            'the movie achieves as great an impact by keeping these thoughts hidden as ... ( quills ) did by showing them . '
        ]
        
        normlime = it.NormLIMENLPInterpreter(self.paddle_model, device='cpu')
        # compute but not save intermediate results.
        normlime.interpret(
            reviews, self.text_to_input_fn, 20, 20,
            unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
            pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]'),
            temp_data_file=None
        )

        # compute and save
        normlime.interpret(
            reviews, self.text_to_input_fn, 20, 20,
            unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
            pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]')
        )

        # load already computed one.
        normlime.interpret(
            reviews, self.text_to_input_fn, 20, 20,
            unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
            pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]')
        )

        os.remove('all_lime_weights.npz')
        os.remove('normlime_weights.npy')
        os.remove('normlime_weights-0.npy')
        os.remove('normlime_weights-1.npy')
