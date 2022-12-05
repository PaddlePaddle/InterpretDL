import warnings
import numpy as np
from collections.abc import Iterable

from ..data_processor.readers import preprocess_image, read_image, restore_image
from ..data_processor.visualizer import sp_weights_to_image_explanation, overlay_threshold, save_image, show_vis_explanation

from ._lime_base import LimeBase
from .abc_interpreter import Interpreter, InputOutputInterpreter


class LIMECVInterpreter(InputOutputInterpreter):
    """
    LIME presents a locally explanation by fitting a set of perturbed samples near the target sample using an 
    interpretable model, specifically a linear model. 

    The implementation is based on https://github.com/marcotcr/lime.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938.
    """

    def __init__(self,
                 model: callable,
                 device: str = 'gpu:0',
                 random_seed: int or None = None):
        """

        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        InputOutputInterpreter.__init__(self, model, device)

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed)
        self.lime_results = {}

    def interpret(self,
                  data: str,
                  interpret_class: int = None,
                  top_k: int = 1,
                  num_samples: int = 1000,
                  batch_size: int = 50,
                  resize_to: int = 224,
                  crop_to: int = None,
                  visual: bool = True,
                  save_path: str = None):
        """
        Main function of the interpreter.

        The implementation is based on https://github.com/marcotcr/lime.

        Args:
            data (str): The input file path.
            interpret_class (int, optional): The index of class to interpret. If None, the most likely label will be 
                used. Default: ``None``.
            top_k (int, optional): Number of top classes to interpret. Will not be used if ``interpret_class`` is 
                given. Default: ``1``.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate 
                interpretation. Default: ``1000``.
            batch_size (int, optional): Number of samples to forward each time. Default: ``50``.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            visual (bool, optional): Whether or not to visualize the processed image. Default: ``True``.
            save_path (str, optional): The filepath(s) to save the processed image(s). If None, the image will not be 
                saved. Default: ``None``.

        Returns:
            [dict]: LIME results: {interpret_label_i: weights on features}
        """
        # preprocess_inputs
        if isinstance(data, str):
            img = read_image(data, resize_to, crop_to)
        else:
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0)
            if np.issubdtype(data.dtype, np.integer):
                img = data
            else:
                # for later visualization
                img = restore_image(data.copy())
        data = preprocess_image(img)
        data_type = np.array(data).dtype
        self.data_type = data_type

        self._build_predict_fn(output='probability')

        probability, _, _ = self.predict_fn(data, None)
        # only one example here
        probability = probability[0]

        if interpret_class is None:
            # only interpret top 1 if not provided.
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-top_k:]
            interpret_class = np.array(interpret_class)
        elif isinstance(interpret_class, list):
            interpret_class = np.array(interpret_class)
        else:
            interpret_class = np.array([interpret_class])

        def predict_fn_for_lime(_imgs):
            _data = preprocess_image(_imgs)  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]

            output, _, _ = self.predict_fn(_data, None)
            return output

        self.predict_fn_for_lime = predict_fn_for_lime
        lime_weights, r2_scores = self.lime_base.interpret_instance(img[0],
                                                                    self.predict_fn_for_lime,
                                                                    interpret_class,
                                                                    num_samples=num_samples,
                                                                    batch_size=batch_size)

        # visualization and save image.
        if save_path is None and not visual:
            # no need to visualize or save explanation results.
            pass
        else:
            explanation_mask = sp_weights_to_image_explanation(img[0], lime_weights, interpret_class[0],
                                                               self.lime_base.segments)
            explanation_vis = overlay_threshold(img[0], explanation_mask)
            if visual:
                show_vis_explanation(explanation_vis)
            if save_path is not None:
                save_image(save_path, explanation_vis)

        # intermediate results, for possible further usages.
        self.lime_results['probability'] = {c: probability[c] for c in interpret_class.ravel()}
        self.lime_results['input'] = img[0]
        self.lime_results['segmentation'] = self.lime_base.segments
        self.lime_results['r2_scores'] = r2_scores
        self.lime_results['lime_weights'] = lime_weights

        return lime_weights


class LIMENLPInterpreter(InputOutputInterpreter):
    """
    LIME Interpreter for NLP tasks.

    LIME presents a locally explanation by fitting a set of perturbed samples near the target sample using an 
    interpretable model, specifically a linear model. 

    The implementation is based on https://github.com/marcotcr/lime.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938.

    """

    def __init__(self,
                 model: callable,
                 device: str = 'gpu:0',
                 random_seed: int or None = None) -> None:
        """

        Args:
            model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
            random_seed (int): random seed. Defaults to None.
        """
        InputOutputInterpreter.__init__(self, model, device)

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed)
        self.lime_results = {}

    def interpret(self,
                  raw_text: str,
                  tokenizer: callable = None,
                  text_to_input_fn: callable = None,
                  preprocess_fn: callable = None,
                  unk_id: int = 0,
                  pad_id: int = 0,
                  classes_to_interpret: list or np.ndarray = None,
                  num_samples: int = 1000,
                  batch_size: int = 50,
                  max_seq_len: int = 128,
                  visual: bool = False):
        """
        Main function of the interpreter.

        The implementation is based on https://github.com/marcotcr/lime.

        Args:
            data (str): The raw string for analysis.
            tokenizer (callable): 
            text_to_input (callable): A user-defined function that convert raw text string to a tuple of inputs 
                that can be fed into the NLP model.
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. 
                Default: ``None``.
            classes_to_interpret (list or numpy.ndarray, optional): The index of class to interpret. If None, the most
                likely label will be used. can be Default: ``None``.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate
                interpretation. Default: ``1000``.
            batch_size (int, optional): Number of samples to forward each time. Default: ``50``.
            visual (bool, optional): Whether or not to visualize. Default: ``True``.

        Returns:
            [dict]: LIME results: {interpret_label_i: weights on features}
        """
        if preprocess_fn is not None:
            text_to_input_fn = preprocess_fn
            warnings.warn('``preprocess_fn`` would be deprecated soon. Use ``text_to_input`` directly.', stacklevel=2)
        assert (tokenizer is None) + (text_to_input_fn is None) == 1, "only one of them should be given."

        # tokenizer to text_to_input.
        if tokenizer is not None:
            if hasattr(tokenizer, 'pad_token_id'):
                pad_id = tokenizer.pad_token_id
                print("According to the tokenizer, pad_token_id is set to", pad_id)
            if hasattr(tokenizer, 'unk_token_id'):
                unk_id = tokenizer.unk_token_id
                print("According to the tokenizer, unk_token_id is set to", unk_id)
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
            self.model_inputs = tuple(inp for inp in model_input)
        else:
            self.model_inputs = tuple(model_input, )

        self._build_predict_fn(output='probability')
        def predict_fn_for_lime(*inputs):
            probability, _, _ = self.predict_fn(inputs, None)
            return probability

        probability, _, _ = self.predict_fn(self.model_inputs, classes_to_interpret)
        # only one example here
        probability = probability[0]

        # only interpret top 1
        if classes_to_interpret is None:
            pred_label = np.argsort(probability)
            classes_to_interpret = pred_label[-1:]

        # this api is from LIME official repo: https://github.com/marcotcr/lime.
        lime_weights, r2_scores = self.lime_base.interpret_instance_text(self.model_inputs,
                                                                         classifier_fn=predict_fn_for_lime,
                                                                         interpret_labels=classes_to_interpret,
                                                                         unk_id=unk_id,
                                                                         pad_id=pad_id,
                                                                         num_samples=num_samples,
                                                                         batch_size=batch_size)

        # intermediate results, for possible further usages.
        self.predicted_proba = probability
        self.lime_results['probability'] = {c: probability[c] for c in classes_to_interpret.ravel()}
        self.lime_results['r2_scores'] = r2_scores
        self.lime_results['lime_weights'] = lime_weights

        if visual:
            # TODO: visualize if tokenizer is given.
            print("Visualization is not supported yet.")
            print("Currently please see the tutorial for the visualization:")
            print("https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb")

        return lime_weights