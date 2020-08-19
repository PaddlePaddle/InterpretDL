import os
import typing
from typing import Any, Callable, List, Tuple, Union
import numpy as np

from ..data_processor.readers import preprocess_image, read_image
from ..data_processor.visualizer import show_important_parts, visualize_image, save_image

from ._lime_base import LimeBase
from .abc_interpreter import Interpreter


class LIMECVInterpreter(Interpreter):
    """
    LIME Interpreter for CV tasks.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self,
                 paddle_model: Callable,
                 trained_model_path: str,
                 model_input_shape=[3, 224, 224],
                 use_cuda=True) -> None:
        """
        Initialize the LIMECVInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                    It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """

        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.model_input_shape = model_input_shape
        self.use_cuda = use_cuda
        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase()

        self.lime_intermediate_results = {}

    def interpret(self,
                  data,
                  interpret_class=None,
                  num_samples=1000,
                  batch_size=50,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (str): The input file path.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be used. Default: None
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The path to save the processed image. If None, the image will not be saved. Default: None

        :return: LIME Prior weights: {interpret_label_i: weights on features}
        :rtype: dict

        Example::

            import interpretdl as it
            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            lime = it.LIMECVInterpreter(paddle_model, "assets/ResNet50_pretrained")
            lime_weights = lime.interpret(
                    'assets/catdog.png',
                    num_samples=1000,
                    batch_size=100,
                    save_path='assets/catdog_lime.png')

        """
        if isinstance(data, str):
            data_instance = read_image(data)
        else:
            data_instance = data

        self.input_type = type(data_instance)
        self.data_type = np.array(data_instance).dtype

        if not self.paddle_prepared:
            self._paddle_prepare()
        # only one example here
        probability = self.predict_fn(data_instance)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        lime_weights, r2_scores = self.lime_base.interpret_instance(
            data_instance[0],
            self.predict_fn,
            interpret_class,
            num_samples=num_samples,
            batch_size=batch_size)

        interpretation = show_important_parts(data_instance[0], lime_weights,
                                              interpret_class[0],
                                              self.lime_base.segments)

        if visual:
            visualize_image(interpretation)

        if save_path is not None:
            save_image(save_path, interpretation)

        self.lime_intermediate_results['probability'] = probability
        self.lime_intermediate_results['input'] = data_instance[0]
        self.lime_intermediate_results[
            'segmentation'] = self.lime_base.segments
        self.lime_intermediate_results['r2_scores'] = r2_scores

        return lime_weights

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
                        dtype='float32')
                    probs = self.paddle_model(data_op)
                    if isinstance(probs, tuple):
                        probs = probs[0]
                    main_program = main_program.clone(for_test=True)

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            self.place = place
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data_instance):
                data = preprocess_image(
                    data_instance
                )  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
                [result] = exe.run(main_program,
                                   fetch_list=[probs],
                                   feed={'data': data})

                return result

        self.predict_fn = predict_fn
        self.paddle_prepared = True


class LIMENLPInterpreter(Interpreter):
    """
    LIME Interpreter for NLP tasks.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self,
                 paddle_model: Callable,
                 trained_model_path: str,
                 use_cuda=True) -> None:
        """
        Initialize the LIMENLPInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                    It takes the following arguments:

                    - data: Data inputs.
                    and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """

        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.use_cuda = use_cuda
        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase()

        self.lime_intermediate_results = {}

    def interpret(self,
                  data,
                  interpret_class=None,
                  num_samples=1000,
                  batch_size=50,
                  unk_id=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (numpy.ndarray or fluid.LoDTensor): The word ids model_input_shape.
            interpret_class (list or numpuy.ndarray, optional): The index of class to interpret. If None, the most likely label will be used. Default: None
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            unk_id (int, optional): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The path to save the processed image. If None, the image will not be saved. Default: None

        :return: LIME Prior weights: {interpret_label_i: weights on features}
        :rtype: dict

        Example::

            import interpretdl as it
            def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
                emb = fluid.embedding(
                    input=data, size=[len(word_dict), EMB_DIM], is_sparse=True)
                conv_3 = fluid.nets.sequence_conv_pool(
                    input=emb,
                    num_filters=hid_dim,
                    filter_size=3,
                    act="tanh",
                    pool_type="sqrt")
                conv_4 = fluid.nets.sequence_conv_pool(
                    input=emb,
                    num_filters=hid_dim,
                    filter_size=4,
                    act="tanh",
                    pool_type="sqrt")
                prediction = fluid.layers.fc(input=[conv_3, conv_4],
                                             size=class_dim,
                                             act="softmax")
                return prediction

            CLASS_DIM = 2
            EMB_DIM = 128
            HID_DIM = 512
            BATCH_SIZE = 128
            print('preparing word_dict...')
            word_dict = paddle.dataset.imdb.word_dict()

            def paddle_model(data):
                probs = convolution_net(data,
                                        len(word_dict), CLASS_DIM, EMB_DIM, HID_DIM)
                return probs

            lime = it.LIMENLPInterpreter(paddle_model, "assets/sent_persistables")

            reviews_str = [b'read the book forget the movie']

            reviews = [c.split() for c in reviews_str]

            UNK = word_dict['<unk>']
            lod = []
            for c in reviews:
                lod.append([word_dict.get(words, UNK) for words in c])

            base_shape = [[len(c) for c in lod]]
            lod = np.array(sum(lod, []), dtype=np.int64)

            data = fluid.create_lod_tensor(lod, base_shape, fluid.CUDAPlace(0))
            print('Begin intepretation...')
            lime_weights = lime.interpret(
                data, num_samples=2000, batch_size=20, unk_id=UNK)

            id2word = dict(zip(word_dict.values(), word_dict.keys()))
            for y in lime_weights:
                print([(id2word[t[0]], t[1]) for t in lime_weights[y]])

        """

        if isinstance(data, np.ndarray) and len(data.shape) == 1:
            data_instance = np.array([data])
        else:
            data_instance = data

        self.input_type = type(data_instance)
        self.data_type = np.array(data_instance).dtype

        if not self.paddle_prepared:
            self._paddle_prepare()
        # only one example here
        probability = self.predict_fn(data_instance)[0]

        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        lime_weights, r2_scores = self.lime_base.interpret_instance_text(
            data_instance,
            self.predict_fn,
            interpret_class,
            unk_id=unk_id,
            num_samples=num_samples,
            batch_size=batch_size)

        data_array = data_instance[0] if isinstance(
            data_instance, np.ndarray) else np.array(data_instance)
        for c in lime_weights:
            weights_c = lime_weights[c]
            weights_new = [(data_array[tup[0]], tup[1]) for tup in weights_c]
            lime_weights[c] = weights_new

        return lime_weights

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():
                    data_op = fluid.data(
                        name='data', shape=[None], dtype='int64', lod_level=1)

                    probs = self.paddle_model(data_op)
                    if isinstance(probs, tuple):
                        probs = probs[0]
                    main_program = main_program.clone(for_test=True)

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            self.place = place
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data_instance):
                if isinstance(data_instance, fluid.LoDTensor):
                    data = data_instance
                else:
                    batch_size, n_features = data_instance.shape
                    samples = np.array(
                        sum(data_instance.tolist(), []), dtype=np.int64)
                    data = fluid.create_lod_tensor(
                        samples, [[n_features] * batch_size], self.place)

                [result] = exe.run(main_program,
                                   fetch_list=[probs],
                                   feed={'data': data})

                return result

        self.predict_fn = predict_fn
        self.paddle_prepared = True
