from __future__ import print_function
import paddle
import paddle.fluid as fluid

import typing
from typing import Any, Callable, List, Tuple, Union

from .abc_interpreter import Interpreter

import IPython.display as display
import cv2
import numpy as np
import paddle.fluid as fluid
import os, sys
from PIL import Image
import pickle


class ForgettingEventsInterpreter(Interpreter):
    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]):
        """
        Initialize the ForgettingEventsInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions. It takes in data inputs and output predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.use_cuda = use_cuda
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

    def interpret(self,
                  train_reader,
                  optimizer,
                  batch_size,
                  epochs,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            train_reader (callable): A training data generator.
            optimizer (paddle.fluid.optimizer): The paddle optimizer.
            batch_size (int): Number of samples to forward each time.
            epochs (int): The number of epochs to train the model.
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None

        Returns:
            numpy.ndarray: ordered_indices, the indices of data in the dataset ordered by count of forgetting events
            numpy.ndarray: ordered_stats, ordered count of forgetting events that corresponds to the indices of data

        Example::

            def conv_bn_layer(input,
                              ch_out,
                              filter_size,
                              stride,
                              padding,
                              act='relu',
                              bias_attr=False):
                tmp = fluid.layers.conv2d(
                    input=input,
                    filter_size=filter_size,
                    num_filters=ch_out,
                    stride=stride,
                    padding=padding,
                    act=None,
                    bias_attr=bias_attr)
                return fluid.layers.batch_norm(input=tmp, act=act)

            def shortcut(input, ch_in, ch_out, stride):
                if ch_in != ch_out:
                    return conv_bn_layer(input, ch_out, 1, stride, 0, None)
                else:
                    return input

            def basicblock(input, ch_in, ch_out, stride):
                tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
                tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
                short = shortcut(input, ch_in, ch_out, stride)
                return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

            def layer_warp(block_func, input, ch_in, ch_out, count, stride):
                tmp = block_func(input, ch_in, ch_out, stride)
                for i in range(1, count):
                    tmp = block_func(tmp, ch_out, ch_out, 1)
                return tmp

            def resnet_cifar10(ipt, depth=32):
                # depth should be one of 20, 32, 44, 56, 110, 1202
                assert (depth - 2) % 6 == 0
                n = (depth - 2) // 6
                nStages = {16, 64, 128}
                conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
                res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
                res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
                res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
                pool = fluid.layers.pool2d(
                    input=res3, pool_size=8, pool_type='avg', pool_stride=1)
                predict = fluid.layers.fc(input=pool, size=10, act='softmax')
                return predict

            def reader_creator(filename, sub_name, cycle=False):
                def read_batch(batch, counter):
                    data = batch[b'data']
                    labels = batch.get(b'labels', batch.get(b'fine_labels', None))
                    assert labels is not None
                    for sample, label in zip(data, labels):
                        global saved
                        if not saved:
                            global samples, sample_labels
                            samples.append(sample)
                            sample_labels.append(label)
                        counter += 1
                        yield counter, (sample / 255.0).astype(np.float32), int(label)

                def reader():
                    global samples, sample_labels
                    counter = 0
                    with tarfile.open(filename, mode='r') as f:
                        names = (each_item.name for each_item in f
                                 if sub_name in each_item.name)
                        while True:
                            for name in names:
                                batch = pickle.load(f.extractfile(name), encoding = 'bytes')
                                for item in read_batch(batch, counter):
                                    yield item
                            if not cycle:
                                break
                        global saved
                        saved = True
                return reader

            fe = ForgettingEventsInterpreter(resnet_cifar10, True, [3, 32, 32])

            samples = []
            sample_labels = []
            saved = False
            paddle.dataset.cifar.reader_creator = reader_creator

            BATCH_SIZE = 128
            train_reader = paddle.batch(paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

            optimizer = fluid.optimizer.Adam(learning_rate=0.001)

            results = fe.interpret(train_reader,
                          optimizer,
                          batch_size = BATCH_SIZE,
                          epochs = 2,
                          save_path = 'assets/test')

            with open("assets/samples.pkl", "wb") as f:
                pickle.dump(np.array(samples), f)
            with open("assets/sample_labels.pkl", "wb") as f:
                pickle.dump(np.array(sample_labels), f)
        """
        stats = {}

        main_program = fluid.default_main_program()
        star_program = fluid.default_startup_program()

        if self.use_cuda:
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            place = fluid.CUDAPlace(gpu_id)
        else:
            place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        avg_cost, probs, label = self._inference()

        optimizer.minimize(avg_cost)

        feed_order = ['data', 'label']

        feed_var_list_loop = [
            main_program.global_block().var(var_name)
            for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)

        for i in range(epochs):
            counter = 0
            correct = 0
            total = 0
            for step_id, data_train in enumerate(train_reader()):
                data_feeded = [t[1:] for t in data_train]
                cost_out, probs_out, label_out = exe.run(
                    main_program,
                    feed=feeder.feed(data_feeded),
                    fetch_list=[avg_cost, probs, label])
                predicted = np.argmax(probs_out, axis=1)
                bsz = len(predicted)
                label_out = label_out.reshape((bsz, ))
                acc = (predicted == label_out).astype(int)
                for k in range(bsz):
                    idx = data_train[k][0]
                    index_stats = stats.get(idx, [])
                    index_stats.append(acc[k])
                    stats[idx] = index_stats

                correct += np.sum(acc)
                total += bsz
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                    (i + 1, epochs, step_id + 1, cost_out.item(),
                     100. * correct / total))
                sys.stdout.flush()

        ordered_indices, ordered_stats = self._compute_and_order_forgetting_stats(
            stats, epochs)
        if save_path is not None:
            with open(save_path + "_ordered_indices.pkl", "wb") as f:
                pickle.dump(ordered_indices, f)
            with open(save_path + "_ordered_stats.pkl", "wb") as f:
                pickle.dump(ordered_stats, f)
        return ordered_indices, ordered_stats

    def _inference(self):

        images = fluid.data(
            name='data',
            shape=[None] + self.model_input_shape,
            dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')

        probs = self.paddle_model(images)
        cost = fluid.layers.cross_entropy(input=probs, label=label)
        avg_cost = fluid.layers.mean(cost)

        return avg_cost, probs, label

    def _compute_and_order_forgetting_stats(self, stats, epochs):
        unlearned_per_presentation = {}
        first_learned = {}

        for example_id, example_stats in stats.items():

            presentation_acc = np.array(example_stats[:epochs])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(transitions
                                                                  == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

        example_original_order = []
        example_stats = []
        for example_id in unlearned_per_presentation:
            example_original_order.append(example_id)

            if np.isnan(first_learned[example_id]):
                example_stats.append(epochs)
            else:
                example_stats.append(
                    len(unlearned_per_presentation[example_id]))

        print('\n Number of unforgettable examples: {}'.format(
            len(np.where(np.array(example_stats) == 0)[0])))

        return np.array(example_original_order)[np.argsort(
            example_stats)], np.sort(example_stats)
