import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import os, sys

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image, extract_img_paths
from ..data_processor.visualizer import visualize_overlay


class IntGradInterpreter(Interpreter):
    """
    Integrated Gradients Interpreter.

    More details regarding the Integrated Gradients method can be found in the original paper:
    http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the IntGradInterpreter.

        Args:
            paddle_model: A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data input.
                - alpha: A scalar for calculating the path integral
                - baseline: The baseline input.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 244, 244]

        Returns:
        """
        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.use_cuda = use_cuda
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

    def interpret(self,
                  data,
                  label=None,
                  baseline=None,
                  steps=50,
                  num_random_trials=10,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (str or numpy.ndarray or fluid.LoDTensor): If task is cv, input can be the image filepath or processed image;
            if task is nlp, input is a fluid.LoDTensor of word ids.
            label (int, optional): The target label to analyze. If None, the most likely label will be used. Default: None
            baseline (str, optional): The baseline input. If None, all zeros will be used. If 'random', random Guassian initialization will be used.
            setps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            num_random_trials (int, optional): number of random initializations to take average in the end. Default: 10
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None

        Returns:
            numpy.ndarray: avg_gradients

        Example::

            def paddle_model(data, alpha, baseline):
                class_num = 1000
                image_input = baseline + alpha * (data - baseline)
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return image_input, probs

            ig = IntGradInterpreter(paddle_model, "assets/ResNet50_pretrained", True)
            gradients = ig.interpret(
                    'assets/catdog.png',
                    label=None,
                    baseline='random',
                    steps=50,
                    num_random_trials=1,
                    visual=True,
                    save_path='ig_test.jpg')
        """

        if isinstance(data, str) or len(np.array(data).shape) > 2:
            task_type = 'cv'
        else:
            task_type = 'nlp'
            baseline = None

        self.task_type = task_type
        self.label = label
        if baseline is None:
            num_random_trials = 1
        is_dir = False

        # Process images
        if task_type == 'cv':
            if isinstance(data, str):
                if os.path.isdir(data):
                    is_dir = True
                    imgs = []
                    img_paths, img_names = extract_img_paths(data)
                    for fp in img_paths:
                        img = read_image(
                            fp, crop_size=self.model_input_shape[1])
                        imgs.append(img)
                    data = np.stack(
                        [preprocess_image(img) for img in imgs], axis=1)[0]
                else:
                    imgs = read_image(
                        data, crop_size=self.model_input_shape[1])
                    data = preprocess_image(imgs)

            else:
                imgs = restore_image(data.copy())

        self.data_type = np.array(data).dtype
        self.input_type = type(data)

        if baseline is None:
            try:
                self.baseline = np.zeros(
                    (num_random_trials, ) + data.shape, dtype=self.data_type)
            except:
                self.baseline = np.zeros((num_random_trials,) + \
                                         (np.sum(data.recursive_sequence_lengths()),), dtype=self.data_type)
        elif baseline == 'random':
            self.baseline = np.random.normal(
                size=(num_random_trials, ) + data.shape).astype(self.data_type)
        else:
            self.baseline = baseline

        if not self.paddle_prepared:
            self._paddle_prepare()

        try:
            n = data.shape[0]
            m = n
        except:
            n = len(data.recursive_sequence_lengths()[0])
            m = 1

        if self.label is None:
            gradients, out, data_out = self.predict_fn(
                data,
                np.array([[0]] * n),
                np.array([[float(1)]] * m), self.baseline[0])
            self.label = np.argmax(out, axis=1)

        gradients_list = []
        for i in range(num_random_trials):
            total_gradients = np.zeros_like(gradients)
            for alpha in np.linspace(0, 1, steps):
                gradients, _, emb = self.predict_fn(data,
                                                    self.label.reshape((n, 1)),
                                                    np.array([[alpha]] * m),
                                                    self.baseline[i])
                total_gradients += np.array(gradients)

            if self.task_type == 'cv':
                ig_gradients = total_gradients * (
                    data_out - self.baseline[i]) / steps
            else:
                ig_gradients = total_gradients * data_out / steps

            gradients_list.append(ig_gradients)
        avg_gradients = np.average(np.array(gradients_list), axis=0)

        # visualize and save the gradients
        if task_type == 'cv':
            if is_dir:
                for i, name in enumerate(img_names):
                    if save_path is None:
                        visualize_overlay([avg_gradients[i]], imgs[i], visual,
                                          save_path)
                    elif os.path.isdir(save_path):
                        visualize_overlay([avg_gradients[i]], imgs[i], visual,
                                          os.path.join(save_path,
                                                       'ig_' + img_names[i]))
            else:
                for i in range(avg_gradients.shape[0]):
                    if os.path.isdir(save_path):
                        visualize_overlay(
                            [avg_gradients[i]], [imgs[i]], visual,
                            os.path.join(save_path, 'ig_%d.jpg' % i))
                    elif save_path is None or isinstance(save_path, str):
                        visualize_overlay([avg_gradients[i]], [imgs[i]],
                                          visual, save_path)

        return avg_gradients

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():

                    if self.input_type == fluid.LoDTensor:
                        data_op = fluid.data(
                            name='data',
                            shape=[None],
                            dtype=self.data_type,
                            lod_level=1)
                        baseline_op = fluid.data(
                            name='baseline',
                            shape=[None],
                            dtype=self.data_type,
                            lod_level=1)
                    else:
                        data_op = fluid.data(
                            name='data',
                            shape=[None] + self.model_input_shape,
                            dtype=self.data_type)
                        baseline_op = fluid.data(
                            name='baseline',
                            shape=[None] + self.model_input_shape,
                            dtype=self.data_type)

                    label_op = fluid.layers.data(
                        name='label', shape=[None, 1], dtype='int64')
                    alpha_op = fluid.layers.data(
                        name='alpha', shape=[None, 1], dtype='double')

                    x_step, probs = self.paddle_model(data_op, alpha_op,
                                                      baseline_op)

                    for op in main_program.global_block().ops:
                        if op.type == 'batch_norm':
                            op._set_attr('use_global_stats', True)
                        elif op.type == 'dropout':
                            op._set_attr('dropout_prob', 0.0)

                    class_num = probs.shape[-1]

                    one_hot = fluid.layers.one_hot(label_op, class_num)
                    one_hot = fluid.layers.elementwise_mul(probs, one_hot)
                    target_category_loss = fluid.layers.reduce_sum(
                        one_hot, dim=1)

                    p_g_list = fluid.backward.append_backward(
                        target_category_loss)

                    gradients_map = fluid.gradients(one_hot, x_step)[0]

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data, label, alpha, baseline):

                gradients, out, emb, loss_out = exe.run(
                    main_program,
                    feed={
                        'data': data,
                        'label': label,
                        'alpha': alpha,
                        'baseline': baseline
                    },
                    fetch_list=[
                        gradients_map, probs, x_step, target_category_loss
                    ],
                    return_numpy=False)

                return gradients, out, emb

        self.predict_fn = predict_fn
        self.paddle_prepared = True
