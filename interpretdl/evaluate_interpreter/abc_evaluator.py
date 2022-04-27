import abc
import sys
import warnings

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class InterpreterEvaluator(ABC):
    """
    InterpreterEvaluator is the base abstract class for all interpreter evaluators. The core function ``evaluate``
    should be implemented.

    All evaluators aim to evaluate the trustworthiness of the interpretation algorithms. Besides theoretical
    verification of the algorithm, here the evaluators validate the trustworthiness by looking through the obtained
    explanations from the interpretation algorithms. Different evaluators are provided.
    
    .. warning:: ``use_cuda`` would be deprecated soon. Use ``device`` directly.
    """

    def __init__(self, paddle_model: callable or None, device: str, use_cuda: bool or None, **kwargs):
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions. This 
                is not always required if the model is not involved. 
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc. Again, this is not always required if the model is not involved.
        """

        if use_cuda in [True, False]:
            warnings.warn('``use_cuda`` would be deprecated soon. '
                          'Use ``device`` directly.', stacklevel=2)
            device = 'gpu' if use_cuda and device[:3] == 'gpu' else 'cpu'

        self.device = device
        self.paddle_model = paddle_model
        self.predict_fn = None

    def evaluate(self, **kwargs):
        raise NotImplementedError
