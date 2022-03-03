import abc
import sys
import warnings

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class InterpreterEvaluator(ABC):
    """Base class for interpreter evaluators.

    Args:
        ABC ([type]): [description]
    """
    def __init__(self, paddle_model: callable, device: str, use_cuda: bool, **kwargs):
        
        if use_cuda in [True, False]:
            warnings.warn(
                '``use_cuda`` would be deprecated soon. '
                'Use ``device`` directly.',
                stacklevel=2
            )
            device = 'gpu' if use_cuda and device[:3] == 'gpu' else 'cpu'

        self.device = device
        self.paddle_model = paddle_model
        self.predict_fn = None
    
    def evaluate(self, **kwargs):
        raise NotImplementedError