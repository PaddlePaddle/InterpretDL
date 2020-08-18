from .lime import LIMECVInterpreter, LIMENLPInterpreter
from .gradient_cam import GradCAMInterpreter
from .integrated_gradients import IntGradCVInterpreter, IntGradNLPInterpreter
from .smooth_grad import SmoothGradInterpreter
from .occlusion import OcclusionInterpreter
from .gradient_shap import GradShapCVInterpreter, GradShapNLPInterpreter
from ._normlime_base import NormLIMECVInterpreter, NormLIMENLPInterpreter
from .score_cam import ScoreCAMInterpreter

__all__ = [
    "LIMECVInterpreter", "LIMENLPInterpreter", "NormLIMECVInterpreter",
    "NormLIMENLPInterpreter", "GradCAMInterpreter", "IntGradCVInterpreter",
    "IntGradNLPInterpreter", "SmoothGradInterpreter", "OcclusionInterpreter",
    "GradShapCVInterpreter", "GradShapNLPInterpreter", "ScoreCAMInterpreter"
]

try:
    import paddle
    from . import lime_prior
    from .lime_prior import LIMEPriorInterpreter
    from .forgetting_events import ForgettingEventsInterpreter
    __all__.append("LIMEPriorInterpreter")
    __all__.append("ForgettingEventsInterpreter")
except ModuleNotFoundError:
    print(
        "Warning: Paddle should be installed before using LIMEPriorInterpreter or ForgettingEventsInterpreter."
    )
