import numpy as np

from .abc_interpreter import Interpreter
from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class LRPCVInterpreter(Interpreter):
    """
    Layer-wise Relevance Propagation Interpreter for CV tasks.

    More details regarding the Layer-wise Relevance Propagation method can be found in the original paper:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=None,
                 device='gpu:0') -> None:
        """

        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu:0``, ``gpu:1`` etc.
            use_cuda (bool):  Would be deprecated soon. Use ``device`` directly.

        """
        Interpreter.__init__(self, paddle_model, device, use_cuda)
        self.paddle_prepared = False

    def interpret(self,
                  inputs,
                  label=None,
                  resize_to=224, 
                  crop_to=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. 
                If None, the most likely label for each  image will be used. Default: None
            resize_to (int, optional): [description]. Images will be rescaled with the shorter edge being `resize_to`. Defaults to 224.
            crop_to ([type], optional): [description]. After resize, images will be center cropped to a square image with the size `crop_to`. 
                If None, no crop will be performed. Defaults to None.
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None
        
        Returns:
            [numpy.ndarray]: interpretations/Relevance map for images.
            
        """
        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)
        
        bsz = len(data)
        save_path = preprocess_save_path(save_path, bsz)

        if not self.paddle_prepared:
            self._paddle_prepare()

        R, output = self.predict_fn(data, label)

        # visualization and save image.
        for i in range(len(imgs)):
            vis_explanation = explanation_to_vis(imgs[i], R[i].squeeze(), style='overlay_grayscale')
            if visual:
                show_vis_explanation(vis_explanation)
            if save_path[i] is not None:
                save_image(save_path[i], vis_explanation)

        return R

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            import paddle
            paddle.set_device(self.device)
            self.paddle_model.eval()

            layer_list = [(n, v) for n, v in self.paddle_model.named_sublayers()]
            num_classes = layer_list[-1][1].weight.shape[1]

            def predict_fn(data, label):
                data = paddle.to_tensor(data, stop_gradient=False)
                output = self.paddle_model(data)

                if label is None:
                    T = output.argmax().numpy()[0]
                else:
                    assert isinstance(label, int), "label should be an integer"
                    assert 0 <= label < num_classes, f"input label is not correct, label should be at [0, {num_classes})"
                    T = label

                T = np.expand_dims(T, 0)
                T = (T[:, np.newaxis] == np.arange(num_classes)) * 1.0
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
