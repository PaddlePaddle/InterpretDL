from __future__ import print_function

import numpy as np
import os, sys
import pickle
import paddle

from .abc_interpreter import Interpreter


class ForgettingEventsInterpreter(Interpreter):
    """
    Forgetting Events Interpreter.
    More details regarding the Forgetting Events method can be found in the original paper:
    https://arxiv.org/pdf/1812.05159.pdf
    """

    def __init__(self, paddle_model, device='gpu:0', use_cuda=None):
        """
        Initialize the ForgettingEventsInterpreter.
        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions. It takes in data inputs and output predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda)

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
        """
        stats = {}
        if save_path is None:
            save_path = 'assets'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        paddle.set_device(self.device)

        for i in range(epochs):
            counter = 0
            correct = 0
            total = 0
            for step_id, data_train in enumerate(train_reader()):
                if isinstance(data_train[0][1], np.ndarray):
                    x_train = [t[1] for t in data_train]
                else:
                    x_train = [t[1].numpy() for t in data_train]
                y_train = [t[2] for t in data_train]
                x_train = paddle.to_tensor(x_train)
                y_train = paddle.to_tensor(np.array(y_train).reshape((-1, 1)))
                logits = self.paddle_model(x_train)
                predicted = paddle.argmax(logits, axis=1).numpy()
                bsz = len(predicted)

                loss = paddle.nn.functional.softmax_with_cross_entropy(logits,
                                                                       y_train)
                avg_loss = paddle.mean(loss)
                y_train = y_train.reshape((bsz, )).numpy()

                acc = (predicted == y_train).astype(int)

                for k in range(bsz):
                    idx = data_train[k][0]
                    # first list is acc, second list is predicted label
                    index_stats = stats.get(idx, [[], []])
                    index_stats[0].append(acc[k])
                    index_stats[1].append(predicted[k])
                    stats[idx] = index_stats

                avg_loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                correct += np.sum(acc)
                total += bsz
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                    (i + 1, epochs, step_id + 1, avg_loss.numpy().item(),
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
        thre = np.mean(scores) + 5 * np.std(scores)

        noisy_pairs = [p for p in pairs if p[1] > thre]
        sorted_noisy_pairs = sorted(
            noisy_pairs, key=lambda x: x[1], reverse=True)
        img_ids = [p[0] for p in sorted_noisy_pairs]

        return img_ids
