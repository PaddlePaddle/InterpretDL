import numpy as np
from .abc_interpreter import Interpreter


class ConsensusInterpreter(object):
    """
    
    Consensus averages the explanations of a given Interpreter over a list of models.
    The averaged result is more like an explanation for the data, instead of specific models.

    More details regarding the Consensus method can be found in the original paper:
    https://arxiv.org/abs/2109.00707.

    """
    
    def __init__(self, InterpreterClass, list_of_models: list, device: str, use_cuda=None, **kwargs):
        """[summary]

        Args:
            InterpreterClass ([type]): The given Interpreter defined in InterpretDL.
            list_of_models (list): a list of model classes. Can be found from paddle.vision.models, or 
                https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/__init__.py. 
            device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu:0``, ``gpu:1`` etc.
        """
        assert issubclass(InterpreterClass, Interpreter)

        self.InterpreterClass = InterpreterClass
        self.list_of_models = list_of_models
        self.device = device
        self.use_cuda = use_cuda
        self.other_args = kwargs

    def interpret(self, inputs: str or list(str) or np.ndarray, **kwargs) -> np.ndarray:
        """Main function of the interpreter.

        We leave the visualization to users. 
        See https://github.com/PaddlePaddle/InterpretDL/tree/master/tutorials/consensus_tutorial_cv.ipynb for an example.

        .. code-block:: python

            import interpretdl as it
            from paddle.vision.models import resnet34, resnet50, resnet101, mobilenet_v2

            list_models = {
                'resnet34': resnet34(pretrained=True), 
                'resnet50': resnet50(pretrained=True),
                'resnet101': resnet101(pretrained=True), 
                'mobilenet_v2': mobilenet_v2(pretrained=True)
            }
            consensus = ConsensusInterpreter(it.SmoothGradInterpreter, list_models.values(), device='gpu:0')

            import matplotlib.pyplot as plt
            import numpy as np

            cols = len(list_models) + 1
            psize = 4
            fig, ax = plt.subplots(1, cols, figsize=(cols*psize, 1*psize))

            for axis in ax:
                axis.axis('off')

            for i in range(len(list_models)):
                ax[i].imshow(np.abs(exp[i]).sum(0))
                ax[i].set_title(list(list_models.keys())[i])

            ax[-1].imshow(np.abs(exp).sum(1).mean(0))
            ax[-1].set_title('Consensus')

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.

        Returns:
            np.ndarray: Concatenated raw explanations.
        """
        
        exps = []
        for model in self.list_of_models:
            interpreter = self.InterpreterClass(model, self.use_cuda, self.device, **self.other_args)
            raw_explanation = interpreter.interpret(inputs, visual=False, save_path=None, **kwargs)
            exps.append(raw_explanation)
        
        return np.concatenate(exps)