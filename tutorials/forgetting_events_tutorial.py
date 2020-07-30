from assets.resnet import ResNet50
import paddle.fluid as fluid
import numpy as np
import sys
import paddle
import paddle.fluid as fluid

sys.path.append('..')
from interpretdl.interpreter.forgetting_events import ForgettingEventsInterpreter
from PIL import Image

import tarfile, pickle, itertools


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
                    batch = pickle.load(f.extractfile(name), encoding='bytes')
                    for item in read_batch(batch, counter):
                        yield item
                if not cycle:
                    break
            global saved
            saved = True

    return reader


if __name__ == '__main__':
    fe = ForgettingEventsInterpreter(resnet_cifar10, True, [3, 32, 32])

    samples = []
    sample_labels = []
    saved = False
    paddle.dataset.cifar.reader_creator = reader_creator

    BATCH_SIZE = 128

    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)

    results = fe.interpret(
        train_reader,
        optimizer,
        batch_size=BATCH_SIZE,
        epochs=2,
        save_path='assets/test__')
    with open("assets/samples.pkl", "wb") as f:
        pickle.dump(np.array(samples), f)
    with open("assets/sample_labels.pkl", "wb") as f:
        pickle.dump(np.array(sample_labels), f)
