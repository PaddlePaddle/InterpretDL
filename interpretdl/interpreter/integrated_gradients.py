import warnings
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable

try:
    from .abc_interpreter_m import InputGradientInterpreter, IntermediateGradientInterpreter
except:
    from .abc_interpreter import InputGradientInterpreter, IntermediateGradientInterpreter

from ..data_processor.readers import images_transform_pipeline, preprocess_save_path
from ..data_processor.visualizer import explanation_to_vis, show_vis_explanation, save_image


class IntGradCVInterpreter(InputGradientInterpreter):
    """
    Integrated Gradients Interpreter for CV tasks.

    For input gradient based interpreters, the target issue is generally the vanilla input gradient's noises.
    The basic idea of reducing the noises is to use different similar inputs to get the input gradients and 
    do the average. 
    
    IntGrad uses the Riemann approximation of the integral, i.e., interpolated values between a baseline (zero) 
    and the original input as inputs, and computes the gradients which will be averaged as the final explanation.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365.
    """

    def __init__(self, model: callable, device: str = 'gpu:0', **kwargs):
        """
        
        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        InputGradientInterpreter.__init__(self, model, device, **kwargs)

    def interpret(self,
                  inputs: str or list(str) or np.ndarray,
                  labels: list or tuple or np.ndarray or None = None,
                  baselines: np.ndarray or None = None,
                  steps: int = 50,
                  num_random_trials: int = 10,
                  gradient_of: str = 'probability',
                  resize_to: int = 224,
                  crop_to: int = None,
                  visual: bool = True,
                  save_path: str = None) -> np.ndarray:
        """The technical details of the IntGrad method are described as follows:
        Given ``inputs``, IntGrad interpolates ``steps`` points between ``baselines`` (usually set to zeros) and 
        ``inputs``. ``baselines`` can be set to ``random``, so that ``num_random_trials`` baselines are used, 
        instead of zeros. Then IntGrad computes the gradients *w.r.t.* these interpolated values and averages the
        results as final explanation.

        Args:
            inputs (str or list): The input image filepath or a list of filepaths or numpy array of read images.
            labels (list or tuple or np.ndarray or None, optional): The target labels to analyze. The number of labels 
                should be equal to the number of images. If None, the most likely label for each image will be used. 
                Default: ``None``.
            baselines (np.ndarray or None, optional): The baseline images to compare with. It should have the same 
                shape as images and same length as the number of images. If None, the baselines of all zeros will be 
                used. Default: ``None``.
            steps (int, optional): number of steps in the Riemann approximation of the integral. Default: ``50``.
            num_random_trials (int, optional): number of random initializations to take average in the end. 
                Default: ``10``.
            gradient_of (str, optional): compute the gradient of ['probability', 'logit' or 'loss']. Default: 
                ``'probability'``. Multi-class classification uses probabitliy, while binary classification uses logit.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The filepath(s) to save the processed image(s). If None, the image will not be 
                saved. Default: ``None``.

        Returns:
            np.ndarray: the explanation result.
        """

        imgs, data = images_transform_pipeline(inputs, resize_to, crop_to)
        bsz = len(data)
        self.data_type = np.array(data).dtype

        self._build_predict_fn(gradient_of=gradient_of)

        if baselines is None:
            num_random_trials = 1
            self.baselines = np.zeros((num_random_trials, ) + data.shape, dtype=self.data_type)
        elif baselines == 'random':
            self.baselines = np.random.normal(size=(num_random_trials, ) + data.shape).astype(self.data_type)
        else:
            self.baselines = baselines

        # obtain the labels (and initialization).
        _, predicted_label, predicted_proba = self.predict_fn(data, labels)
        self.predicted_label = predicted_label
        self.predicted_proba = predicted_proba
        if labels is None:
            labels = predicted_label

        labels = np.array(labels).reshape((bsz, ))

        # IntGrad.
        gradients_list = []
        with tqdm(total=num_random_trials * steps, leave=True, position=0) as pbar:
            for i in range(num_random_trials):
                total_gradients = np.zeros_like(data)
                for alpha in np.linspace(0, 1, steps):
                    data_scaled = data * alpha + self.baselines[i] * (1 - alpha)
                    gradients, _, _ = self.predict_fn(data_scaled, labels)
                    total_gradients += gradients
                    pbar.update(1)

                ig_gradients = total_gradients * (data - self.baselines[i]) / steps
                gradients_list.append(ig_gradients)
        avg_gradients = np.average(np.array(gradients_list), axis=0)

        # visualization and save image.
        if save_path is None and not visual:
            # no need to visualize or save explanation results.
            pass
        else:
            save_path = preprocess_save_path(save_path, bsz)
            for i in range(bsz):
                vis_explanation = explanation_to_vis(imgs[i],
                                                     np.abs(avg_gradients[i]).sum(0),
                                                     style='overlay_grayscale')
                if visual:
                    show_vis_explanation(vis_explanation)
                if save_path[i] is not None:
                    save_image(save_path[i], vis_explanation)

        return avg_gradients


class IntGradNLPInterpreter(IntermediateGradientInterpreter):
    """
    Integrated Gradients Interpreter for NLP tasks.
        
    For input gradient based interpreters, the target issue is generally the vanilla input gradient's noises.
    The basic idea of reducing the noises is to use different similar inputs to get the input gradients and 
    do the average. 

    The inputs for NLP tasks are considered as the embedding features. So the noises or the changes of inputs
    are done for the embeddings.

    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365.
    """

    def __init__(self, model: callable, device: str = 'gpu:0', **kwargs) -> None:
        """
        
        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        IntermediateGradientInterpreter.__init__(self, model, device, **kwargs)

    def interpret(self,
                  raw_text: str,
                  tokenizer: callable = None,
                  text_to_input_fn: callable = None,
                  label: list or np.ndarray = None,
                  steps: int = 50,
                  gradient_of: str = 'logit',
                  embedding_name: str = 'word_embeddings',
                  max_seq_len: int = 128,
                  visual: bool = False) -> np.ndarray:
        """The technical details of the IntGrad method for NLP tasks are similar for CV tasks, except the noises are
        added on the embeddings.

        Args:
            data (tupleornp.ndarray): The inputs to the NLP model.
            labels (listornp.ndarray, optional): The target labels to analyze. If None, the most likely label 
                will be used. Default: ``None``.
            steps (int, optional): number of steps in the Riemann approximation of the integral. Default: ``50``.
            gradient_of (str, optional): compute the gradient of ['probability', 'logit' or 'loss']. Default: 
                ``'logit'``. Multi-class classification uses probabitliy, while binary classification uses logit.
            embedding_name (str, optional): name of the embedding layer at which the noises will be applied. 
                The name of embedding can be verified through ``print(model)``. Defaults to ``word_embeddings``. 

        Returns:
            np.ndarray or tuple: explanations, or (explanations, pred).
        """
        assert (tokenizer is None) + (text_to_input_fn is None) == 1, "only one of them should be given."

        # tokenizer to text_to_input.
        if tokenizer is not None:
            def text_to_input_fn(raw_text):
                encoded_inputs = tokenizer(text=raw_text, max_seq_len=max_seq_len)
                # order is important. *_batched_and_to_tuple will be the input for the model.
                _batched_and_to_tuple = tuple([np.array([v]) for v in encoded_inputs.values()])
                return _batched_and_to_tuple
        else:
            print("Warning: Visualization can not be supported if tokenizer is not given.")

        # from raw text string to token ids (and other terms that the user-defined function outputs).
        model_input = text_to_input_fn(raw_text)
        if isinstance(model_input, Iterable) and not hasattr(model_input, 'shape'):
            model_input = tuple(inp for inp in model_input)
        else:
            model_input = tuple(model_input, )

        # TODO: layer_name to be matched using re.
        self._build_predict_fn(layer_name=embedding_name, gradient_of=gradient_of)

        gradients, label, data_out, proba = self.predict_fn(model_input, label, scale=None)

        # IG
        total_gradients = np.zeros_like(gradients)
        for alpha in np.linspace(0, 1, steps):
            gradients, _, _, _ = self.predict_fn(model_input, label, scale=alpha)
            total_gradients += gradients

        ig_gradients = total_gradients * data_out / steps

        # intermediate results, for possible further usages.
        self.predicted_label = label
        self.predicted_proba = proba

        if visual:
            # TODO: visualize if tokenizer is given.
            print("Visualization is not supported yet.")
            print("Currently please see the tutorial for the visualization:")
            print("https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb")

        return ig_gradients
