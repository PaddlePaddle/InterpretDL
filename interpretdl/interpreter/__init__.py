# Files within this path, contain the interpretation algorithms.

from .abc_interpreter import Interpreter
from .abc_interpreter import InputGradientInterpreter, InputOutputInterpreter, IntermediateLayerInterpreter, TransformerInterpreter

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
from .generic_attention import GAInterpreter, GANLPInterpreter, GACVInterpreter
from .bidirectional_transformer import BTCVInterpreter, BTNLPInterpreter

__all__ = [
    "Interpreter", "InputGradientInterpreter", "InputOutputInterpreter", "IntermediateLayerInterpreter", "TransformerInterpreter",
    "LIMECVInterpreter", "LIMENLPInterpreter", "GradCAMInterpreter", "IntGradCVInterpreter", "IntGradNLPInterpreter",
    "SmoothGradInterpreter", "OcclusionInterpreter", "GradShapCVInterpreter", "GradShapNLPInterpreter",
    "ScoreCAMInterpreter", "LRPCVInterpreter", "RolloutInterpreter", "TAMInterpreter", "SmoothGradInterpreterV2",
    "ConsensusInterpreter", "GAInterpreter", "BTCVInterpreter", "BTNLPInterpreter", "GANLPInterpreter", "GACVInterpreter"
]

try:
    import paddle
    from .lime_prior import LIMEPriorInterpreter
    from .glime import GLIMECVInterpreter
    from .forgetting_events import ForgettingEventsInterpreter
    from ._normlime_base import NormLIMECVInterpreter, NormLIMENLPInterpreter
    from .training_dynamics import TrainingDynamics, BHDFInterpreter
    __all__ += [
        "LIMEPriorInterpreter", "GLIMECVInterpreter", "ForgettingEventsInterpreter", "NormLIMECVInterpreter",
        "NormLIMENLPInterpreter", "BHDFInterpreter","TrainingDynamics"
    ]
except ModuleNotFoundError:
    print("Warning: Paddle should be installed before using some Interpreters.")
