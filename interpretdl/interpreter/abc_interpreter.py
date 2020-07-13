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

    def _paddle_prepare(self):
        """
        Prepare Paddle programs.
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