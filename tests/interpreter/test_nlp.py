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
        from tutorials.assets.utils import convert_example, aggregate_subwords_and_importances
        
        MODEL_NAME = "ernie-2.0-en"
        model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)
        tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

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

    def test_lime_intgrad_gradshap_nlp(self):
        self.prepare()

        reviews = [
            "it 's a charming and often affecting journey . ",
        ]
        data = [ {"text": r} for r in reviews]

        algo = it.LIMENLPInterpreter(self.paddle_model, device='cpu')
        for i, review in enumerate(data):
            pred_class, pred_prob, lime_weights = algo.interpret(
                review,
                self.preprocess_fn,
                num_samples=22,
                batch_size=20,
                unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
                pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]'),
                return_pred=True)

        algo = it.IntGradNLPInterpreter(self.paddle_model, device='cpu')
        pred_labels, pred_probs, avg_gradients = algo.interpret(
            self.preprocess_fn(data),
            steps=2,
            return_pred=True)

        algo = it.GradShapNLPInterpreter(self.paddle_model, device='cpu')

        pred_labels, pred_probs, avg_gradients = algo.interpret(
            self.preprocess_fn(data),
            n_samples=2,
            noise_amount=0.1,
            return_pred=True)
            
    def test_normlime(self):
        self.prepare()

        reviews = [
            "it 's a charming and often affecting journey . ",
            'the movie achieves as great an impact by keeping these thoughts hidden as ... ( quills ) did by showing them . '
        ]

        data = [ {"text": r} for r in reviews]
        
        normlime = it.NormLIMENLPInterpreter(self.paddle_model, device='cpu')
        # compute but not save intermediate results.
        normlime.interpret(
            data, self.preprocess_fn, 20, 20,
            unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
            pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]'),
            temp_data_file=None
        )

        # compute and save
        normlime.interpret(
            data, self.preprocess_fn, 20, 20,
            unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
            pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]')
        )

        # load already computed one.
        normlime.interpret(
            data, self.preprocess_fn, 20, 20,
            unk_id=self.tokenizer.convert_tokens_to_ids('[UNK]'),
            pad_id=self.tokenizer.convert_tokens_to_ids('[PAD]')
        )

        os.remove('all_lime_weights.npz')
        os.remove('normlime_weights.npy')
        os.remove('normlime_weights-0.npy')
        os.remove('normlime_weights-1.npy')
