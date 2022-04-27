import numpy as np
from .abc_interpreter import Interpreter


class ConsensusInterpreter(object):
    """
    
    ConsensusInterpreter averages the explanations of a given Interpreter over a list of models. The averaged result 
    is more like an explanation for the data, instead of specific models. For visual object recognition tasks, the 
    Consensus explanation would be more aligned with the object than individual models.

    More details regarding the Consensus method can be found in the original paper:
    https://arxiv.org/abs/2109.00707.

    For reference, the ``list_of_models`` can be found from :py:mod:`paddle.vision.models` or 
    `PPClas <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/__init__.py>`_.
    """

    def __init__(self, InterpreterClass, list_of_models: list, device: str = 'gpu:0', use_cuda=None, **kwargs):
        """
        
        Args:
            InterpreterClass ([type]): The given Interpreter defined in InterpretDL.
            list_of_models (list): a list of trained models.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        assert issubclass(InterpreterClass, Interpreter)

        self.InterpreterClass = InterpreterClass
        self.list_of_models = list_of_models
        self.device = device
        self.use_cuda = use_cuda
        self.other_args = kwargs

    def interpret(self, inputs: str or list(str) or np.ndarray, **kwargs) -> np.ndarray:
        """
        The technical details are simple to understand for the Consensus method:
        Given the ``inputs`` and the interpretation algorithm (one of the Interpreters), each model in 
        ``list_of_models`` will produce an explanation, then Consensus will concatenate all the explanations. 
        Subsequent normalization and average can be done as users' preference. The suggested operation for input
        gradient based algorithms is average of the absolute values.

        We leave the visualization to users. 
        See the `notebook example 
        <https://github.com/PaddlePaddle/InterpretDL/tree/master/tutorials/example_consensus_cv.ipynb>`_ 
        for an example.

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
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy 
                array of read images.

        Returns:
            np.ndarray: Concatenated raw explanations.
        """

        exps = []
        for model in self.list_of_models:
            interpreter = self.InterpreterClass(model, self.device, self.use_cuda, **self.other_args)
            raw_explanation = interpreter.interpret(inputs, visual=False, save_path=None, **kwargs)
            exps.append(raw_explanation)

        return np.concatenate(exps)
