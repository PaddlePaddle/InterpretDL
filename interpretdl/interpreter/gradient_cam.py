import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import os, sys
from PIL import Image

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_image, read_image, restore_image, preprocess_inputs
from ..data_processor.visualizer import visualize_heatmap


class GradCAMInterpreter(Interpreter):
    """
    Gradient CAM Interpreter.

    More details regarding the GradCAM method can be found in the original paper:
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradCAMInterpreter.

        Args:
            paddle_model (callable): A user-defined function that gives access to model predictions.
                It takes the following arguments:

                - data: Data inputs.
                and outputs predictions. See the example at the end of ``interpret()``.
            trained_model_path (str): The pretrained model directory.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self)
        self.paddle_model = paddle_model
        self.trained_model_path = trained_model_path
        self.use_cuda = use_cuda
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

    def interpret(self,
                  inputs,
                  target_layer_name,
                  labels=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            data (str or numpy.ndarray): The input image filepath or numpy array.
            target_layer_name (str): The target layer to calculate gradients.
            label (int, optional): The target label to analyze. If None, the most likely label will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str, optional): The filepath to save the processed image. If None, the image will not be saved. Default: None

        Returns:
            None

        Example::

            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            gradcam = GradCAMInterpreter(paddle_model, "assets/ResNet50_pretrained",True)
            gradcam.interpret(
                    'assets/catdog.png',
                    'res5c.add.output.5.tmp_0',
                    label=None,
                    visual=True,
                    save_path='gradcam_test.jpg')
        """

        imgs, data, save_path = preprocess_inputs(inputs, save_path,
                                                  self.model_input_shape)

        self.target_layer_name = target_layer_name

        if not self.paddle_prepared:
            self._paddle_prepare()

        bsz = len(data)
        if labels is None:
            _, _, out = self.predict_fn(
                data, np.zeros(
                    (bsz, 1), dtype='int64'))
            labels = np.argmax(out, axis=1)
        labels = np.array(labels).reshape((bsz, 1))

        print(labels)

        feature_map, gradients, _ = self.predict_fn(data, labels)

        f = np.array(feature_map)
        g = np.array(gradients)
        mean_g = np.mean(g, (2, 3))
        heatmap = f.transpose([0, 2, 3, 1])
        dim_array = np.ones((1, heatmap.ndim), int).ravel()
        dim_array[heatmap.ndim - 1] = -1
        dim_array[0] = bsz
        heatmap = heatmap * mean_g.reshape(dim_array)

        heatmap = np.mean(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap, axis=tuple(np.arange(1, heatmap.ndim)))
        heatmap /= heatmap_max.reshape((bsz, ) + (1, ) * (heatmap.ndim - 1))
        for i in range(bsz):
            visualize_heatmap(heatmap[i], imgs[i], visual, save_path[i])

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle.fluid as fluid
            startup_prog = fluid.Program()
            main_program = fluid.Program()
            with fluid.program_guard(main_program, startup_prog):
                with fluid.unique_name.guard():

                    image_op = fluid.data(
                        name='image',
                        shape=[None] + self.model_input_shape,
                        dtype='float32')
                    label_op = fluid.layers.data(
                        name='label', shape=[None, 1], dtype='int64')

                    probs = self.paddle_model(image_op)
                    if isinstance(probs, tuple):
                        probs = probs[0]

                    # manually switch the model to test mode
                    for op in main_program.global_block().ops:
                        if op.type == 'batch_norm':
                            op._set_attr('use_global_stats', True)
                        elif op.type == 'dropout':
                            op._set_attr('dropout_prob', 0.0)

                    # fetch the target layer
                    trainable_vars = list(main_program.list_vars())
                    for v in trainable_vars:
                        if v.name == self.target_layer_name:
                            conv = v

                    class_num = probs.shape[-1]
                    one_hot = fluid.layers.one_hot(label_op, class_num)
                    one_hot = fluid.layers.elementwise_mul(probs, one_hot)
                    target_category_loss = fluid.layers.reduce_sum(
                        one_hot, dim=1)
                    # target_category_loss = - fluid.layers.cross_entropy(probs, label_op)[0]

                    # add back-propagration
                    p_g_list = fluid.backward.append_backward(
                        target_category_loss)
                    # calculate the gradients w.r.t. the target layer
                    gradients_map = fluid.gradients(target_category_loss,
                                                    conv)[0]

            if self.use_cuda:
                gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                place = fluid.CUDAPlace(gpu_id)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            fluid.io.load_persistables(exe, self.trained_model_path,
                                       main_program)

            def predict_fn(data, labels):
                # if label is None, let it be the most likely label
                #if self.label is None:
                #    out = exe.run(
                #        main_program,
                #        feed={'image': data,
                #              'label': np.array([[0]])},
                #        fetch_list=[probs])

                #    self.label = np.argmax(out[0][0])

                feature_map, gradients, out = exe.run(
                    main_program,
                    feed={'image': data,
                          'label': labels},
                    fetch_list=[conv, gradients_map, probs])
                return feature_map, gradients, out

        self.predict_fn = predict_fn
        self.paddle_prepared = True
