# Files within this path, contain the interpretation algorithms.

from .abc_interpreter import Interpreter
from .abc_interpreter import InputGradientInterpreter, InputOutputInterpreter, IntermediateLayerInterpreter

from .lime import LIMECVInterpreter, LIMENLPInterpreter
from .gradient_cam import GradCAMInterpreter
from .integrated_gradients import IntGradCVInterpreter, IntGradNLPInterpreter
from .smooth_grad import SmoothGradInterpreter
from .smooth_grad_v2 import SmoothGradInterpreterV2
from .occlusion import OcclusionInterpreter
from .gradient_shap import GradShapCVInterpreter, GradShapNLPInterpreter
from .score_cam import ScoreCAMInterpreter
from .lrp import LRPCVInterpreter
from .rollout import RolloutInterpreter
from .transition_attention_maps import TAMInterpreter
from .consensus import ConsensusInterpreter

__all__ = [
    "Interpreter",
    "InputGradientInterpreter", "InputOutputInterpreter", "IntermediateLayerInterpreter",
    "LIMECVInterpreter", "LIMENLPInterpreter", "GradCAMInterpreter", "IntGradCVInterpreter",
    "IntGradNLPInterpreter", "SmoothGradInterpreter", "OcclusionInterpreter",
    "GradShapCVInterpreter", "GradShapNLPInterpreter", "ScoreCAMInterpreter",
    "LRPCVInterpreter", "RolloutInterpreter", "TAMInterpreter",
    "SmoothGradInterpreterV2", 'ConsensusInterpreter'
]

try:
    import paddle
    from . import lime_prior
    from .lime_prior import LIMEPriorInterpreter
    from .forgetting_events import ForgettingEventsInterpreter
    from ._normlime_base import NormLIMECVInterpreter, NormLIMENLPInterpreter
    __all__ += ["LIMEPriorInterpreter", "ForgettingEventsInterpreter", 
                "NormLIMECVInterpreter", "NormLIMENLPInterpreter"]
except ModuleNotFoundError:
    print(
        "Warning: Paddle should be installed before using some Interpreters."
    )
