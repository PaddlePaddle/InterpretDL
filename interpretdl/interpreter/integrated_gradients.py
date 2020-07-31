import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import os, sys

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image
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
            data (str or numpy.ndarray): If task is cv, input can be the image filepath or processed image; if task is nlp, input a sequence of word ids.
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
                import paddle.fluid as fluid
                class_num = 1000
                image_input = baseline + alpha * data
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
            input_type = 'cv'
            self.baseline = baseline
        else:
            input_type = 'nlp'
            self.baseline = None

        self.label = label
        self.num_random_trials = num_random_trials
        self.steps = steps

        # Read in image
        if isinstance(data, str):
            img = read_image(data, crop_size=self.model_input_shape[1])
            data = preprocess_image(img)

        else:
            img = restore_image(data.copy())

        self.data_type = np.array(data).dtype

        if not self.paddle_prepared:
            self._paddle_prepare()

        avg_gradients = self.predict_fn(data)

        if input_type == 'cv':
            visualize_overlay(avg_gradients, img, visual, save_path)

        return avg_gradients

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():

                    data_op = fluid.data(
                        name='data',
                        shape=[1] + self.model_input_shape,
                        dtype=self.data_type)
                    label_op = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')
                    alpha_op = fluid.layers.data(
                        name='alpha', shape=[1], dtype='double')

                    if self.baseline == 'random':
                        x_baseline = fluid.layers.gaussian_random(
                            [1] + self.model_input_shape, dtype=self.data_type)
                    else:
                        x_baseline = fluid.layers.zeros_like(data_op)

                    x_diff = data_op - x_baseline

                    x_step, probs = self.paddle_model(x_diff, alpha_op,
                                                      x_baseline)

                    for op in main_program.global_block().ops:
                        if op.type == 'batch_norm':
                            op._set_attr('use_global_stats', True)
                        elif op.type == 'dropout':
                            op._set_attr('dropout_prob', 0.0)

                    class_num = probs.shape[-1]

                    one_hot = fluid.layers.one_hot(label_op, class_num)
                    one_hot = fluid.layers.elementwise_mul(probs, one_hot)
                    target_category_loss = fluid.layers.reduce_sum(one_hot)

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

            def predict_fn(data):
                gradients, out, data_diff = exe.run(
                    main_program,
                    feed={
                        'data': data,
                        'label': np.array([[0]]),
                        'alpha': np.array([[float(1)]]),
                    },
                    fetch_list=[gradients_map, probs, x_step],
                    return_numpy=False)

                # if label is None, let it be the most likely label
                if self.label is None:
                    self.label = np.argmax(out[0])

                gradients_list = []

                if self.baseline is None:
                    num_random_trials = 1

                for i in range(self.num_random_trials):
                    total_gradients = np.zeros_like(gradients)
                    for alpha in np.linspace(0, 1, self.steps):
                        [gradients] = exe.run(main_program,
                                              feed={
                                                  'data': data,
                                                  'label':
                                                  np.array([[self.label]]),
                                                  'alpha': np.array([[alpha]]),
                                              },
                                              fetch_list=[gradients_map],
                                              return_numpy=False)
                        total_gradients += np.array(gradients)
                    ig_gradients = total_gradients * np.array(
                        data_diff) / self.steps
                    gradients_list.append(ig_gradients)
                avg_gradients = np.average(np.array(gradients_list), axis=0)
                return avg_gradients

        self.predict_fn = predict_fn
        self.paddle_prepared = True
