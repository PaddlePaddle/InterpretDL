from assets.resnet import ResNet50
import paddle.fluid as fluid
import numpy as np
import sys
import paddle
import paddle.fluid as fluid

sys.path.append('..')
from interpretdl.interpreter.forgetting_events import ForgettingEventsInterpreter
from PIL import Image
import matplotlib.pyplot as plt
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


def reader_prepare(data, labels):
    def reader():
        counter_ = -1
        for sample, label in zip(data, labels):
            counter_ += 1
            yield counter_, (sample / 255.0).astype(np.float32), int(label)

    return reader


if __name__ == '__main__':
    CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
    filename = paddle.dataset.common.download(CIFAR10_URL, 'cifar',
                                              CIFAR10_MD5)

    all_data = []
    all_labels = []
    with tarfile.open(filename, mode='r') as f:
        names = (each_item.name for each_item in f
                 if 'data_batch' in each_item.name)
        for name in names:
            batch = pickle.load(f.extractfile(name), encoding='bytes')
            data = batch[b'data']
            labels = batch.get(b'labels', batch.get(b'fine_labels', None))
            all_data.extend(data)
            all_labels.extend(labels)

    with open("assets/samples.pkl", "wb") as f:
        pickle.dump(np.array(all_data), f)
    with open("assets/sample_labels.pkl", "wb") as f:
        pickle.dump(np.array(all_labels), f)

    fe = ForgettingEventsInterpreter(resnet_cifar10, True, [3, 32, 32])

    BATCH_SIZE = 128

    train_reader = paddle.batch(
        reader_prepare(all_data, all_labels), batch_size=BATCH_SIZE)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    epochs = 100
    print('Training %d epochs. This may take some time.' % epochs)
    count_forgotten, forgotten = fe.interpret(
        train_reader,
        optimizer,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        save_path='assets/test_')

    print([
        '0 - airplance', '1 - automobile', '2 - bird', '3 - cat', '4 - deer',
        '5 - dog', '6 - frog', '7 - horse', '8 - ship', '9 - truck'
    ])

    max_count = max(count_forgotten.keys())
    max_count_n = len(count_forgotten[max_count])

    show_n = 9
    count = 0
    fig = plt.figure(figsize=(12, 12))
    axes = []
    print('The most frequently forgotten samples: ')
    for k in np.sort(np.array(list(count_forgotten.keys())))[::-1]:
        for idx, i in enumerate(count_forgotten[k][:show_n - count]):
            x = all_data[i].reshape((3, 32, 32)).transpose((1, 2, 0))
            axes.append(fig.add_subplot(3, 3, idx + count + 1))
            axes[-1].set_title(
                'Forgotten {} times, \n True label: {}, Learned as: {}'.format(
                    k, all_labels[i], np.unique(forgotten[i])))
            axes[-1].axis('off')
            plt.imshow(x)
        count += len(count_forgotten[k][:show_n - count])
        if count >= show_n:
            break
    plt.show()

    axes = []
    fig = plt.figure(figsize=(12, 12))
    zero_count_n = len(count_forgotten.get(0, []))
    print('Number of never forgotten samples is %d.' % (zero_count_n))
    for idx, i in enumerate(count_forgotten.get(0, [])[:show_n]):
        x = all_data[i].reshape((3, 32, 32)).transpose((1, 2, 0))
        axes.append(fig.add_subplot(3, 3, idx + 1))
        axes[-1].set_title('label {}'.format(all_labels[i]))
        axes[-1].axis('off')
        plt.imshow(x)
    plt.show()
