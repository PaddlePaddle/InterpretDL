
import numpy as np
import paddle
import re

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class RolloutInterpreter(Interpreter):
    """
    Rollout Interpreter.

    More details regarding the Rollout method can be found in the original paper:
    https://arxiv.org/abs/2005.00928
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the RolloutInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
        """
        Interpreter.__init__(self, paddle_model, 'gpu:0', use_cuda)
        self.paddle_model = paddle_model
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

        self.use_cuda = use_cuda
        if not paddle.is_compiled_with_cuda():
            self.use_cuda = False

        # init for usages during the interpretation.
        self.block_attns = []

    def interpret(self,
                  inputs,
                  start_layer=0,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/heatmap for each image
        :rtype: numpy.ndarray
        """

        imgs, data = preprocess_inputs(inputs, self.model_input_shape)
        bsz = len(data)  # batch size
        save_path = preprocess_save_path(save_path, bsz)

        if not self.paddle_prepared:
            self._paddle_prepare()

        blk_attns, preds = self.predict_fn(data)

        assert start_layer < len(blk_attns), "start_layer should be in the range of [0, num_block-1]"

        label = np.array(preds).reshape((bsz,))

        all_layer_attentions = []
        for attn_heads in blk_attns:
            avg_heads = (attn_heads.sum(axis=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)

        # compute rollout between attention layers
        # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
        num_tokens = all_layer_attentions[0].shape[1]
        batch_size = all_layer_attentions[0].shape[0]
        eye = paddle.eye(num_tokens).expand((batch_size, num_tokens, num_tokens))
        all_layer_attentions = [all_layer_attentions[i] + eye for i in range(len(all_layer_attentions))]
        matrices_aug = [all_layer_attentions[i] / all_layer_attentions[i].sum(axis=-1, keepdim=True)
                            for i in range(len(all_layer_attentions))]
        joint_attention = matrices_aug[start_layer]
        for i in range(start_layer+1, len(matrices_aug)):
            joint_attention = matrices_aug[i].bmm(joint_attention)

        # rollout
        rollout_explanation = joint_attention[:, 0, 1:].reshape((-1, 14, 14)).cpu().detach().numpy()

        # visualization and save image.
        for i in range(bsz):
            vis_explanation = explanation_to_vis(imgs[i], rollout_explanation[i], style='overlay_heatmap')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return rollout_explanation

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            paddle.set_device('gpu:0' if self.use_cuda else 'cpu')
            # to get gradients, the ``train`` mode must be set.
            # we cannot set v.training = False for the same reason.
            self.paddle_model.train()

            # def hook for attention layer
            def hook(layer, input, output):
                self.block_attns.append(output)

            for n, v in self.paddle_model.named_sublayers():
                if re.match('^blocks.*.attn.attn_drop$', n):
                    # print(n)
                    v.register_forward_post_hook(hook)

                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0

                # Report issues or pull requests if more layers need to be added.

            def predict_fn(data):
                data = paddle.to_tensor(data)
                data.stop_gradient = False
                self.block_attns = []
                out = self.paddle_model(data)
                out = paddle.nn.functional.softmax(out, axis=1)
                preds = paddle.argmax(out, axis=1)
                label = preds.numpy()

                return self.block_attns, label

        self.predict_fn = predict_fn
        self.paddle_prepared = True
