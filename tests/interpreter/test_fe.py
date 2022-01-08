import unittest
import paddle
import numpy as np
import shutil
from paddle.vision.models.resnet import resnet50

import interpretdl as it
from tests.utils import assert_arrays_almost_equal


class TestForgettingEvent(unittest.TestCase):

    def test_algo(self):
        from paddle.vision.transforms import Transpose
        from paddle.vision.datasets import Cifar10

        train_dataset = Cifar10(mode='train', transform=Transpose())
        train_dataset = paddle.io.Subset(dataset=train_dataset, indices=list(range(100)))
        paddle_model = resnet50(pretrained=False, num_classes=10)
        def reader_prepare(dataset):
            def reader():
                counter_ = -1
                for sample, label in dataset:
                    counter_ += 1
                    yield counter_, (sample / 255.0).astype(np.float32), int(label)

            return reader

        BATCH_SIZE = 64
        train_reader = paddle.batch(
            reader_prepare(train_dataset), batch_size=BATCH_SIZE)
        optimizer = paddle.optimizer.Momentum(learning_rate=0.001,
                            momentum=0.9,
                            parameters=paddle_model.parameters())        
        epochs = 2
        fe = it.ForgettingEventsInterpreter(paddle_model, 'cpu')
        stats, (count_forgotten, forgotten) = fe.interpret(
            train_reader,
            optimizer,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            save_path='tmp')

        fe.find_noisy_labels(stats)
        shutil.rmtree('tmp')


if __name__ == '__main__':
    unittest.main()