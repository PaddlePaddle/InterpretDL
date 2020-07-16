import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Interpreter(ABC):
    """Interpreter is the base class for all interpretation algorithms.

    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """

    def _paddle_prepare(self, predict_fn=None):
        """
        Prepare Paddle program inside of the interpreter. This will be called by interpret().
        Should not be called explicitly.

        Args:
            predict_fn: A defined callable function that defines inputs and outputs.
                Defaults to None, and each interpreter will generate it.
                example for LIME:
                    def predict_fn(image_input):
                        import paddle.fluid as fluid
                        class_num = 1000
                        model = ResNet101()
                        logits = model.net(input=image_input, class_dim=class_num)
                        probs = fluid.layers.softmax(logits, axis=-1)
                        return probs

        Returns:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def interpret(self, **kwargs):
        """
        Main function of the interpreter.

        :param kwargs:
        :return:
        """
        raise NotImplementedError
