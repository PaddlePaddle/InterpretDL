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

    def _build_predict_fn(self, rebuild: bool = False):
        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        if self.predict_fn is None or rebuild:
            import paddle
            if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
                print("Paddle is not installed with GPU support. Change to CPU version now.")
                self.device = 'cpu'

            # set device. self.device is one of ['cpu', 'gpu:0', 'gpu:1', ...]
            paddle.set_device(self.device)

            # to get gradients, the ``train`` mode must be set.
            self.paddle_model.eval()

            def predict_fn(inputs):
                """predict_fn for input gradients based interpreters,
                    for image classification models only.

                Args:
                    data ([type]): [description]

                Returns:
                    [type]: [description]
                """
                # assert len(inputs.shape) == 4  # [bs, h, w, 3]

                with paddle.no_grad():
                    inputs = tuple(paddle.to_tensor(inp) for inp in inputs) if isinstance(inputs, tuple) \
                        else (paddle.to_tensor(inputs), )
                    logits = self.paddle_model(*inputs)  # get logits, [bs, num_c]
                    probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
                return probas.numpy()

            self.predict_fn = predict_fn

    def evaluate(self, **kwargs):
        raise NotImplementedError
