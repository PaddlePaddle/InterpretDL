import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import os, sys

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image, extract_img_paths
from ..data_processor.visualizer import visualize_overlay


class IntGradCVInterpreter(Interpreter):
    """
    Integrated Gradients Interpreter for CV tasks.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the IntGradCVInterpreter.

        Args:
            paddle_model: A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data input.
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
            data (str or numpy.ndarray): The image filepath or processed images.
            label (list or numpy.ndarray, optional): The target label to analyze. If None, the most likely label will be used. Default: None
            baseline (str or numpy.ndarray, optional): The baseline input. If None, all zeros will be used. If 'random', random Guassian initialization will be used.
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            num_random_trials (int, optional): number of random initializations to take average in the end. Default: 10
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None

        :return: avg_gradients
        :rtype: numpy.ndarray

        Example::

            import interpretdl as it
            import io

            def load_vocab(file_path):
                vocab = {}
                with io.open(file_path, 'r', encoding='utf8') as f:
                    wid = 0
                    for line in f:
                        if line.strip() not in vocab:
                            vocab[line.strip()] = wid
                            wid += 1
                vocab["<unk>"] = len(vocab)
                return vocab

            def paddle_model(data, alpha):
                dict_dim = 1256606
                emb_dim = 128
                # embedding layer
                emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
                emb *= alpha
                probs = bilstm_net_emb(emb, None, None, dict_dim, is_prediction=True)
                return emb, probs

            ig = it.IntGradNLPInterpreter(
                paddle_model, "assets/senta_model/bilstm_model/params", True)

            word_dict = load_vocab("assets/senta_model/bilstm_model/word_dict.txt")
            unk_id = word_dict["<unk>"]
            reviews = [[
                '交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'
            ]]

            lod = []
            for c in reviews:
                lod.append([word_dict.get(words, unk_id) for words in c])
            base_shape = [[len(c) for c in lod]]
            lod = np.array(sum(lod, []), dtype=np.int64)
            data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())

            avg_gradients = ig.interpret(
                data, label=None, steps=50, visual=True, save_path='ig_test.jpg')

            sum_gradients = np.sum(avg_gradients, axis=1).tolist()
            lod = data.lod()

            new_array = []
            for i in range(len(lod[0]) - 1):
                new_array.append(
                    dict(zip(reviews[i], sum_gradients[lod[0][i]:lod[0][i + 1]])))
        """

        self.label = label

        if baseline is None:
            num_random_trials = 1
        is_dir = False

        # Process images
        if isinstance(data, str):
            if os.path.isdir(data):
                is_dir = True
                imgs = []
                img_paths, img_names = extract_img_paths(data)
                for fp in img_paths:
                    _, img = read_image(
                        fp, crop_size=self.model_input_shape[1])
                    imgs.append(img)
                data = np.stack(
                    [preprocess_image(img) for img in imgs], axis=1)[0]
            else:
                _, imgs = read_image(data, crop_size=self.model_input_shape[1])
                data = preprocess_image(imgs)
        else:
            imgs = restore_image(data.copy())

        self.data_type = np.array(data).dtype
        self.input_type = type(data)

        if baseline is None:
            self.baseline = np.zeros(
                (num_random_trials, ) + data.shape, dtype=self.data_type)
        elif baseline == 'random':
            self.baseline = np.random.normal(
                size=(num_random_trials, ) + data.shape).astype(self.data_type)
        else:
            self.baseline = baseline

        if not self.paddle_prepared:
            self._paddle_prepare()

        n = data.shape[0]

        gradients, out = self.predict_fn(data, np.array([[0]] * n))

        if self.label is None:
            self.label = np.argmax(out, axis=1)
        else:
            self.label = np.array(self.label)
        gradients_list = []
        for i in range(num_random_trials):
            total_gradients = np.zeros_like(gradients)
            for alpha in np.linspace(0, 1, steps):
                data_scaled = data * alpha + self.baseline[i] * (1 - alpha)
                gradients, _ = self.predict_fn(data_scaled,
                                               self.label.reshape((n, 1)))
                total_gradients += np.array(gradients)
            ig_gradients = total_gradients * (data - self.baseline[i]) / steps
            gradients_list.append(ig_gradients)
        avg_gradients = np.average(np.array(gradients_list), axis=0)

        # visualize and save the gradients
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
                    visualize_overlay([avg_gradients[i]], [imgs[i]], visual,
                                      os.path.join(save_path, 'ig_%d.jpg' % i))
                elif save_path is None or isinstance(save_path, str):
                    visualize_overlay([avg_gradients[i]], [imgs[i]], visual,
                                      save_path)

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
                        shape=[None] + self.model_input_shape,
                        dtype=self.data_type)

                    label_op = fluid.layers.data(
                        name='label', shape=[None, 1], dtype='int64')

                    data_plus = data_op + fluid.layers.zeros_like(data_op)
                    probs = self.paddle_model(data_plus)

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

                    gradients_map = fluid.gradients(one_hot, data_plus)[0]

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data, label):

                gradients, out = exe.run(main_program,
                                         feed={'data': data,
                                               'label': label},
                                         fetch_list=[gradients_map, probs],
                                         return_numpy=False)

                return gradients, out

        self.predict_fn = predict_fn
        self.paddle_prepared = True


class IntGradNLPInterpreter(Interpreter):
    """
    Integrated Gradients Interpreter for NLP tasks.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self, paddle_model, trained_model_path,
                 use_cuda=True) -> None:
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
        self.paddle_prepared = False

    def interpret(self,
                  data,
                  label=None,
                  steps=50,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (fluid.LoDTensor): The word ids input.
            label (list, optional): The target label to analyze. If None, the most likely label will be used. Default: None
            baseline (str or numpy.ndarray, optional): The baseline input. If None, all zeros will be used. If 'random', random Guassian initialization will be used.
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 50
            num_random_trials (int, optional): number of random initializations to take average in the end. Default: 10
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None

        :return: avg_gradients
        :rtype: numpy.ndarray

        Example::

            import interpretdl as it
            def paddle_model(data):
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=data, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            ig = it.IntGradCVInterpreter(paddle_model, "assets/ResNet50_pretrained", True)
            gradients = ig.interpret(
                    'assets/catdog.png',
                    label=None,
                    baseline='random',
                    steps=50,
                    num_random_trials=1,
                    visual=True,
                    save_path='ig_test.jpg')
        """

        self.label = label

        self.data_type = np.array(data).dtype

        if not self.paddle_prepared:
            self._paddle_prepare()

        n = len(data.recursive_sequence_lengths()[0])

        gradients, out, data_out = self.predict_fn(data,
                                                   np.array([[0]] * n),
                                                   np.array([[float(1)]]))
        if self.label is None:
            self.label = np.argmax(out, axis=1)
        else:
            self.label = np.array(label)

        total_gradients = np.zeros_like(gradients)
        for alpha in np.linspace(0, 1, steps):
            gradients, _, emb = self.predict_fn(data,
                                                self.label.reshape((n, 1)),
                                                np.array([[alpha]]))
            total_gradients += np.array(gradients)

        ig_gradients = total_gradients * data_out / steps

        #avg_gradients = np.average(np.array(gradients_list), axis=0)

        return ig_gradients

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():

                    data_op = fluid.data(
                        name='data', shape=[None], dtype='int64', lod_level=1)

                    label_op = fluid.layers.data(
                        name='label', shape=[None, 1], dtype='int64')
                    alpha_op = fluid.layers.data(
                        name='alpha', shape=[None, 1], dtype='double')

                    x_step, probs = self.paddle_model(data_op, alpha_op)

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

            def predict_fn(data, label, alpha):

                gradients, out, emb = exe.run(
                    main_program,
                    feed={'data': data,
                          'label': label,
                          'alpha': alpha},
                    fetch_list=[gradients_map, probs, x_step],
                    return_numpy=False)

                return gradients, out, emb

        self.predict_fn = predict_fn
        self.paddle_prepared = True
