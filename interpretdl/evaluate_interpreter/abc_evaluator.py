import abc
import sys
import numpy as np
import warnings


# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


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
            if use_cuda and device[:3] == 'gpu':
                device = device
            else:
                device = 'cpu'
        self.device = device
        self.paddle_model = paddle_model
        self.predict_fn = None
    
    def evaluate(self, **kwargs):
        raise NotImplementedError