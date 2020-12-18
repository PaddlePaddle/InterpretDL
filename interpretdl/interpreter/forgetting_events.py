from __future__ import print_function

import numpy as np
import os, sys
import pickle
import paddle
import paddle.fluid as fluid

import typing
from typing import Any, Callable, List, Tuple, Union

from .abc_interpreter import Interpreter


class ForgettingEventsInterpreter(Interpreter):
    """
    Forgetting Events Interpreter.
    More details regarding the Forgetting Events method can be found in the original paper:
    https://arxiv.org/pdf/1812.05159.pdf
    """

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
                  noisy_labels=False,
                  save_path=None):
        """
        Main function of the interpreter.
        Args:
            train_reader (callable): A training data generator.
            optimizer (paddle.fluid.optimizer): The paddle optimizer.
            batch_size (int): Number of samples to forward each time.
            epochs (int): The number of epochs to train the model.
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None
        :return: (count_forgotten, forgotten) where count_forgotten is {count of forgetting events: list of data indices with such count of forgetting events} and forgotten is {data index: numpy.ndarray of wrong predictions that follow true predictions in the training process}
        :rtype: (dict, dict)
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
            def reader_prepare(data, labels):
                def reader():
                    counter_ = -1
                    for sample, label in zip(data, labels):
                        counter_ += 1
                        yield counter_, (sample / 255.0).astype(np.float32), int(label)
                return reader
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
                        'Forgotten {} times, True label: {}, Learned as: {}'.format(
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
        """
        stats = {}
        if save_path is None:
            save_path = 'assets'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        main_program = fluid.default_main_program()
        star_program = fluid.default_startup_program()

        if self.use_cuda:
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            place = fluid.CUDAPlace(gpu_id)
        else:
            place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        avg_cost, probs, label = self._forward()

        test_program = main_program.clone(for_test=True)

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
                    # first list is acc, second list is predicted label
                    index_stats = stats.get(idx, [[], []])
                    index_stats[0].append(acc[k])
                    index_stats[1].append(predicted[k])
                    stats[idx] = index_stats

                correct += np.sum(acc)
                total += bsz
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                    (i + 1, epochs, step_id + 1, cost_out.item(),
                     100. * correct / total))
                sys.stdout.flush()

        with open(os.path.join(save_path, "stats.pkl"), "wb") as f:
            pickle.dump(stats, f)

        if noisy_labels:
            noisy_samples = self.find_noisy_labels(stats)
            return stats, noisy_samples
        else:
            count_forgotten, forgotten = self.compute_and_order_forgetting_stats(
                stats, epochs, save_path)
            return stats, (count_forgotten, forgotten)

    def _forward(self):

        images = fluid.data(
            name='data',
            shape=[None] + self.model_input_shape,
            dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')

        probs = self.paddle_model(images)
        cost = fluid.layers.cross_entropy(input=probs, label=label)
        avg_cost = fluid.layers.mean(cost)

        return avg_cost, probs, label

    def compute_and_order_forgetting_stats(self, stats, epochs,
                                           save_path=None):
        unlearned_per_presentation = {}
        first_learned = {}
        forgotten = {}

        for example_id, example_stats in stats.items():

            # accuracies
            presentation_acc = np.array(example_stats[0][:epochs])
            # predicted labels
            presentation_predicted = np.array(example_stats[1][:epochs])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            if len(np.where(transitions == -1)[0]) > 0:
                # forgotten epochs
                unlearned_per_presentation[example_id] = np.where(transitions
                                                                  == -1)[0] + 2
                # forgotten indices
                forgotten[example_id] = presentation_predicted[np.where(
                    transitions == -1)[0] + 1]

            else:
                unlearned_per_presentation[example_id] = []
                forgotten[example_id] = np.array([])

            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan
                forgotten[example_id] = presentation_predicted

        count_forgotten = {}

        for example_id, forgotten_epochs in unlearned_per_presentation.items():
            if np.isnan(first_learned[example_id]):
                count = -1
            else:
                count = len(forgotten_epochs)

            count_stats = count_forgotten.get(count, [])
            count_stats.append(example_id)
            count_forgotten[count] = count_stats

        if save_path is not None:
            with open(os.path.join(save_path, "count_forgotten.pkl"),
                      "wb") as f:
                pickle.dump(count_forgotten, f)
            with open(os.path.join(save_path, "forgotten.pkl"), "wb") as f:
                pickle.dump(forgotten, f)

        return count_forgotten, forgotten

    def find_noisy_labels(self, stats):
        pairs = []
        for example_id, example_stats in stats.items():
            presentation_acc = np.array(example_stats[0])
            if len(np.where(presentation_acc == 1)[0]) == 0:
                continue
            pairs.append(
                [example_id, np.where(presentation_acc == 1)[0].mean()])

        if len(pairs) == 0:
            return []

        scores = [p[1] for p in pairs]
        thre = np.mean(scores) + 3 * np.std(scores)

        noisy_pairs = [p for p in pairs if p[1] > thre]
        sorted_noisy_pairs = sorted(
            noisy_pairs, key=lambda x: x[1], reverse=True)
        img_ids = [p[0] for p in sorted_noisy_pairs]

        return img_ids
