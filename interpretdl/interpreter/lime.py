
import numpy as np
from numpy.lib.arraysetops import isin
import paddle

from ..data_processor.readers import preprocess_inputs, preprocess_image, read_image, restore_image
from ..data_processor.visualizer import sp_weights_to_image_explanation, overlay_threshold, save_image, show_vis_explanation

from ._lime_base import LimeBase
from .abc_interpreter import Interpreter, InputOutputInterpreter


class LIMECVInterpreter(InputOutputInterpreter):
    """
    LIME Interpreter for CV tasks.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self,
                 paddle_model,
                 use_cuda=None,
                 device='gpu:0',
                 model_input_shape=[3, 224, 224],
                 random_seed=None) -> None:
        """
        Initialize the LIMECVInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """
        InputOutputInterpreter.__init__(self, paddle_model, device, use_cuda)
        self.model_input_shape = model_input_shape

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed)
        self.lime_results = {}

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

        """
        # preprocess_inputs
        if isinstance(data, str):
            crop_size = self.model_input_shape[1]
            target_size = int(self.model_input_shape[1] * 1.143)
            img = read_image(data, target_size, crop_size)
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

        probability, _ = self.predict_fn(data, None)
        # only one example here
        probability = probability[0]

        if interpret_class is None:
            # only interpret top 1 if not provided.
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]
            interpret_class = np.array(interpret_class)
        elif isinstance(interpret_class, list):
            interpret_class = np.array(interpret_class)
        else:
            interpret_class = np.array([interpret_class])
        
        def predict_fn_for_lime(_imgs):
            _data = preprocess_image(
                _imgs
            )  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
            
            output, _ = self.predict_fn(_data, None)
            return output

        self.predict_fn_for_lime = predict_fn_for_lime
        lime_weights, r2_scores = self.lime_base.interpret_instance(
            img[0],
            self.predict_fn_for_lime,
            interpret_class,
            num_samples=num_samples,
            batch_size=batch_size
        )

        # visualization and save image.
        if save_path is None and not visual:
            # no need to visualize or save explanation results.
            pass
        else:
            explanation_mask = sp_weights_to_image_explanation(
                img[0], lime_weights, interpret_class[0], self.lime_base.segments
            )
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


class LIMENLPInterpreter(Interpreter):
    """
    LIME Interpreter for NLP tasks.

    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self, paddle_model, use_cuda=True, random_seed=None) -> None:
        """
        Initialize the LIMENLPInterpreter.

        Args:
            paddle_model (callable): A paddle model that outputs predictions.
            trained_model_path (str): The pretrained model directory.
            model_input_shape (list, optional): The input shape of the model. Default: [3, 224, 224]
            use_cuda (bool, optional): Whether or not to use cuda. Default: True
        """

        Interpreter.__init__(self, paddle_model, 'gpu:0', use_cuda)
        self.paddle_model = paddle_model
        self.use_cuda = use_cuda
        if not paddle.is_compiled_with_cuda():
            self.use_cuda = False

        self.paddle_prepared = False

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed)

        self.lime_intermediate_results = {}

    def interpret(self,
                  data,
                  preprocess_fn,
                  unk_id,
                  pad_id=None,
                  interpret_class=None,
                  num_samples=1000,
                  batch_size=50,
                  lod_levels=None,
                  return_pred=False,
                  visual=True):
        """
        Main function of the interpreter.

        Args:
            data (str): The raw string for analysis.
            preprocess_fn (Callable): A user-defined function that input raw string and outputs the a tuple of inputs to feed into the NLP model.
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. Default: None.
            interpret_class (list or numpy.ndarray, optional): The index of class to interpret. If None, the most likely label will be used. Default: None
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            lod_levels (list or tuple or numpy.ndarray or None, optional): The lod levels for model inputs. It should have the length equal to number of outputs given by preprocess_fn.
                                            If None, lod levels are all zeros. Default: None.
            visual (bool, optional): Whether or not to visualize. Default: True

        :return: LIME Prior weights: {interpret_label_i: weights on features}
        :rtype: dict
        """

        model_inputs = preprocess_fn(data)
        if not isinstance(model_inputs, tuple):
            self.model_inputs = (np.array(model_inputs), )
        else:
            self.model_inputs = tuple(inp.numpy() for inp in model_inputs)

        if not self.paddle_prepared:
            self._paddle_prepare()
        # only one example here
        probability = self.predict_fn(*self.model_inputs)[0]
        probability = paddle.nn.functional.softmax(paddle.to_tensor(probability)).numpy()
        
        # only interpret top 1
        if interpret_class is None:
            pred_label = np.argsort(probability)
            interpret_class = pred_label[-1:]

        lime_weights, r2_scores = self.lime_base.interpret_instance_text(
            self.model_inputs,
            classifier_fn=self.predict_fn,
            interpret_labels=interpret_class,
            unk_id=unk_id,
            pad_id=pad_id,
            num_samples=num_samples,
            batch_size=batch_size)

        data_array = self.model_inputs[0]
        data_array = data_array.reshape((np.prod(data_array.shape), ))
        for c in lime_weights:
            weights_c = lime_weights[c]
            weights_new = [(data_array[tup[0]], tup[1]) for tup in weights_c]
            lime_weights[c] = weights_new

        if return_pred:
            return (interpret_class, probability[interpret_class],
                    lime_weights)
        return lime_weights

    def _paddle_prepare(self, predict_fn=None):
        if predict_fn is None:
            paddle.set_device('gpu:0' if self.use_cuda else 'cpu')
            self.paddle_model.eval()

            def predict_fn(*params):
                params = tuple(paddle.to_tensor(inp) for inp in params)
                probs = self.paddle_model(*params)
                return probs.numpy()

        self.predict_fn = predict_fn
        self.paddle_prepared = True
