import paddle
import numpy as np

from .abc_interpreter import Interpreter
from ..data_processor.readers import preprocess_inputs, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class LRPCVInterpreter(Interpreter):
    """
    Layer-wise Relevance Propagation Interpreter for CV tasks.

    More details regarding the Layer-wise Relevance Propagation method can be found in the original paper:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        """
        Initialize the LRPCVInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]

        """
        super(Interpreter, self).__init__()
        self.paddle_model = paddle_model
        self.model_input_shape = model_input_shape
        self.paddle_prepared = False

        self.use_cuda = use_cuda
        if not paddle.is_compiled_with_cuda():
            self.use_cuda = False

    def interpret(self,
                  inputs,
                  label=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/Relevance map for each image
        :rtype: numpy.ndarray
        """
        imgs, data = preprocess_inputs(inputs, self.model_input_shape)

        if not self.paddle_prepared:
            self._paddle_prepare()

        R, output = self.predict_fn(data, label)

        # visualization and save image.
        vis_explanation = explanation_to_vis(
            imgs, R.squeeze(), style='overlay_grayscale')
        if visual:
            show_vis_explanation(vis_explanation)
        if save_path is not None:
            save_image(save_path, vis_explanation)

        return R

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            paddle.set_device('gpu:0' if self.use_cuda else 'cpu')
            self.paddle_model.eval()

            def predict_fn(data, label):
                data = paddle.to_tensor(data, stop_gradient=False)
                output = self.paddle_model(data)

                if label is None:
                    T = output.argmax().numpy()[0]
                else:
                    assert isinstance(label, int), "label should be an integer"
                    assert 0 <= label < 1000, "input label is not correct, label should be at [0 ,1000)"
                    T = label

                T = np.expand_dims(T, 0)
                T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
                T = paddle.to_tensor(T).astype('float32')

                R = self.paddle_model.relprop(
                    R=output * T, alpha=1).sum(axis=1, keepdim=True)

                # Check relevance value preserved
                # print("Check relevance value preserved: ")
                # print('Pred logit : ' + str((output * T).sum().cpu().numpy()))
                # print('Relevance Sum : ' + str(R.sum().cpu().numpy()))

                return R.detach().numpy(), output.detach().numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True

        return predict_fn
