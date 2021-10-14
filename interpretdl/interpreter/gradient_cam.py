
import numpy as np
import paddle

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class GradCAMInterpreter(Interpreter):
    """
    Gradient CAM Interpreter.

    More details regarding the GradCAM method can be found in the original paper:
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the GradCAMInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self, paddle, 'gpu:0', use_cuda)
        self.paddle_model = paddle_model
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

        self.use_cuda = use_cuda
        if not paddle.is_compiled_with_cuda():
            self.use_cuda = False

        # init for usages during the interpretation.
        self._target_layer_name = None
        self._feature_maps = {}

    def interpret(self,
                  inputs,
                  target_layer_name,
                  label=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            target_layer_name (str): The target layer to calculate gradients.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/heatmap for each image
        :rtype: numpy.ndarray
        """

        imgs, data = preprocess_inputs(inputs, self.model_input_shape)
        bsz = len(data)  # batch size
        save_path = preprocess_save_path(save_path, bsz)

        assert target_layer_name in [n for n, v in self.paddle_model.named_sublayers()], \
            f"target_layer_name {target_layer_name} does not exist in the given model, " \
            f"please check all valid layer names by [n for n, v in paddle_model.named_sublayers()]"
        
        if self._target_layer_name != target_layer_name:
            self._target_layer_name = target_layer_name
            self.paddle_prepared = False

        if not self.paddle_prepared:
            self._paddle_prepare()

        feature_map, gradients, preds = self.predict_fn(data, label)
        if label is None:
            label = preds

        label = np.array(label).reshape((bsz,))
        f = np.array(feature_map.numpy())
        g = gradients

        # print(f.shape, g.shape)  # [bsz, channels, w, h]

        cam_weights = np.mean(g, (2, 3), keepdims=True)
        heatmap = cam_weights * f
        heatmap = heatmap.mean(1)
        # relu
        gradcam_explanation = np.maximum(heatmap, 0)

        # visualization and save image.
        for i in range(bsz):
            vis_explanation = explanation_to_vis(imgs[i], gradcam_explanation[i], style='overlay_heatmap')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return gradcam_explanation

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            paddle.set_device('gpu:0' if self.use_cuda else 'cpu')
            # to get gradients, the ``train`` mode must be set.
            # we cannot set v.training = False for the same reason.
            self.paddle_model.train()

            def hook(layer, input, output):
                self._feature_maps[layer._layer_name_for_hook] = output

            for n, v in self.paddle_model.named_sublayers():
                if n == self._target_layer_name:
                    v._layer_name_for_hook = n
                    v.register_forward_post_hook(hook)
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0
                # Report issues or pull requests if more layers need to be added.

            def predict_fn(data, label):
                data = paddle.to_tensor(data)
                data.stop_gradient = False
                out = self.paddle_model(data)
                out = paddle.nn.functional.softmax(out, axis=1)
                preds = paddle.argmax(out, axis=1)
                if label is None:
                    label = preds.numpy()
                label_onehot = paddle.nn.functional.one_hot(
                    paddle.to_tensor(label), num_classes=out.shape[1])
                target = paddle.sum(out * label_onehot, axis=1)

                target.backward()
                gradients = self._feature_maps[self._target_layer_name].grad
                target.clear_gradient()

                # gradients = paddle.grad(
                #     outputs=[target], inputs=[self._feature_maps])[0]
                if isinstance(gradients, paddle.Tensor):
                    gradients = gradients.numpy()

                return self._feature_maps[self._target_layer_name], gradients, label

        self.predict_fn = predict_fn
        self.paddle_prepared = True
