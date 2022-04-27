import abc
import sys
import numpy as np
import warnings

from ..common.python_utils import versiontuple2tuple

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class Interpreter(ABC):
    """
    Interpreter is the base abstract class for all Interpreters. 
    The implementation of any Interpreter should at least 

        **(1)** prepare :py:attr:`predict_fn` that outputs probability predictions, gradients or other desired 
        intermediate results of the model, and 

        **(2)** implement the core function :py:meth:`interpret` of the interpretation algorithm.
    In general, we find this implementation is practical, makes the code more readable and can highlight the core 
    function of the interpretation algorithm.

    This kind of implementation works for all post-poc interpretation algorithms. While some algorithms may have 
    different features and other fashions of implementations may be more suitable for them, our style of implementation
    can still work for most of them. So we follow this design for all Interpreters in this library.
    
    Three sub-abstract Interpreters that implement :py:meth:`_build_predict_fn` are currently provided in this file:
    :class:`InputGradientInterpreter`, :class:`InputOutputInterpreter`, :class:`IntermediateLayerInterpreter`. For each
    of them, the implemented :py:attr:`predict_fn` can be used by several different algorithms. Therefore, the further 
    implementations can focus on the core algorithm. More sub-abstract Interpreters will be provided if necessary.

    .. warning:: ``use_cuda`` would be deprecated soon. Use ``device`` directly.
    """

    def __init__(self, paddle_model: callable, device: str, use_cuda: bool = None, **kwargs):
        """
        
        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        self.device = device
        self.paddle_model = paddle_model
        self.predict_fn = None

        if use_cuda in [True, False]:
            warnings.warn('``use_cuda`` would be deprecated soon. Use ``device`` directly.', stacklevel=2)
            self.device = 'gpu:0' if use_cuda and device[:3] == 'gpu' else 'cpu'

        assert self.device[:3] in ['cpu', 'gpu']

    def _paddle_prepare(self, predict_fn: callable or None = None):
        """
        Prepare Paddle program inside of the interpreter. This will be called by :py:meth:`interpret`. Would be renamed
        to :py:meth:`_build_predict_fn`.

        Args:
            predict_fn: A defined callable function that defines inputs and outputs. Defaults to ``None``, and each 
                interpreter should implement it.
        """
        raise NotImplementedError

    def interpret(self, **kwargs):
        """Main function of the interpreter."""
        raise NotImplementedError

    def _build_predict_fn(self, **kwargs):
        """ Build :py:attr:`predict_fn` for interpreters. This will be called by :py:meth:`interpret`. """
        raise NotImplementedError

    def _paddle_env_setup(self):
        """Prepare the environment setup. This is not always necessary because the setup can be done within the 
        function of :py:func:`_build_predict_fn`.
        """
        #######################################################################
        # This is a simple implementation for disabling gradient computation. #
        #######################################################################
        import paddle
        if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
            print("Paddle is not installed with GPU support. Change to CPU version now.")
            self.device = 'cpu'

        # globally set device.
        paddle.set_device(self.device)

        # does not need gradients at all.
        self.paddle_model.eval()


class InputGradientInterpreter(Interpreter):
    """This is one of the sub-abstract Interpreters. 
    
    :class:`InputGradientInterpreter` are used by input gradient based Interpreters. Interpreters that are derived from 
    :class:`InputGradientInterpreter` include :class:`GradShapCVInterpreter`, :class:`IntGradCVInterpreter`, 
    :class:`SmoothGradInterpreter`.

    This Interpreter implements :py:func:`_build_predict_fn` that returns input gradient given an input. 
    """

    def __init__(self, paddle_model: callable, device: str, use_cuda: bool = None, **kwargs):
        """
        
        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda, **kwargs)
        assert hasattr(paddle_model, 'forward'), \
            "paddle_model has to be " \
            "an instance of paddle.nn.Layer or a compatible one."

    def _paddle_env_setup(self):
        """Prepare the environment setup. 
        This function is an implementation for gradient computation.
        """
        import paddle
        if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
            print("Paddle is not installed with GPU support. Change to CPU version now.")
            self.device = 'cpu'

        # globally set device.
        paddle.set_device(self.device)
        if versiontuple2tuple(paddle.__version__) >= (2, 2, 1):
            # From Paddle2.2.1, gradients are supported in eval mode.
            self.paddle_model.eval()
        else:
            # Former versions.
            self.paddle_model.train()
            for n, v in self.paddle_model.named_sublayers():
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0

    def _build_predict_fn(self, rebuild: bool = False, gradient_of: str = 'probability'):
        """Build ``predict_fn`` for input gradients based algorithms.
        The model is supposed to be a classification model.

        Args:
            rebuild (bool, optional): forces to rebuild. Defaults to ``False``.
            gradient_of (str, optional): computes the gradient of 
                [``"loss"``, ``"logit"`` or ``"probability"``] *w.r.t.* input data. Defaults to ``"probability"``. 
                Other options can get similar results while the absolute scale might be different.
        """

        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        if self.predict_fn is None or rebuild:
            assert gradient_of in ['loss', 'logit', 'probability']

            self._paddle_env_setup()

            def predict_fn(data, labels=None):
                """predict_fn for input gradients based interpreters,
                    for image classification models only.

                Args:
                    data ([type]): scaled input data.
                    labels ([type]): can be None.

                Returns:
                    [type]: gradients, labels
                """
                import paddle
                assert len(data.shape) == 4  # [bs, h, w, 3]
                assert labels is None or \
                    (isinstance(labels, (list, np.ndarray)) and len(labels) == data.shape[0])

                data = paddle.to_tensor(data)
                data.stop_gradient = False

                # get logits, [bs, num_c]
                logits = self.paddle_model(data)
                num_samples, num_classes = logits.shape[0], logits.shape[1]

                # get predictions.
                pred = paddle.argmax(logits, axis=1)
                if labels is None:
                    labels = pred.numpy()

                # get gradients
                if gradient_of == 'loss':
                    # cross-entropy loss
                    loss = paddle.nn.functional.cross_entropy(logits, paddle.to_tensor(labels), reduction='sum')
                else:
                    # logits or probas
                    labels = np.array(labels).reshape((num_samples, ))
                    labels_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(labels), num_classes=num_classes)
                    if gradient_of == 'logit':
                        loss = paddle.sum(logits * labels_onehot, axis=1)
                    else:
                        probas = paddle.nn.functional.softmax(logits, axis=1)
                        loss = paddle.sum(probas * labels_onehot, axis=1)

                loss.backward()
                gradients = data.grad
                if isinstance(gradients, paddle.Tensor):
                    gradients = gradients.numpy()

                return gradients, labels

            self.predict_fn = predict_fn


class InputOutputInterpreter(Interpreter):
    """This is one of the sub-abstract Interpreters. 
    
    :class:`InputOutputInterpreter` are used by input-output correlation based Interpreters. Interpreters that are derived
    from :class:`InputOutputInterpreter` include :class:`OcclusionInterpreter`, :class:`LIMECVInterpreter`, 
    :class:`SmoothGradInterpreter`.

    This Interpreter implements :py:func:`_build_predict_fn` that returns the model's prediction given an input. 

    """

    def __init__(self, paddle_model: callable, device: str, use_cuda: bool = None, **kwargs):
        """
        
        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """
        Interpreter.__init__(self, paddle_model, device, use_cuda, **kwargs)
        assert hasattr(paddle_model, 'forward'), \
            "paddle_model has to be " \
            "an instance of paddle.nn.Layer or a compatible one."

    def _build_predict_fn(self, rebuild: bool = False, output: str = 'probability'):
        """Build :py:attr:`predict_fn` for Input-Output based algorithms.
        The model is supposed to be a classification model.

        Args:
            rebuild (bool, optional): forces to rebuild. Defaults to ``False``.
            output (str, optional): computes the logit or probability. Defaults: ``"probability"``. Other options can 
                get similar results while the absolute scale might be different.
        """

        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        if self.predict_fn is None or rebuild:
            assert output in ['logit', 'probability']

            self._paddle_env_setup()

            def predict_fn(data, label):
                """predict_fn for input gradients based interpreters,
                    for image classification models only.

                Args:
                    data ([type]): [description]
                    label ([type]): can be None.

                Returns:
                    [type]: [description]
                """
                import paddle
                assert len(data.shape) == 4  # [bs, h, w, 3]

                with paddle.no_grad():
                    logits = self.paddle_model(paddle.to_tensor(data))  # get logits, [bs, num_c]
                    probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
                    pred = paddle.argmax(probas, axis=1)  # get predictions.

                    if label is None:
                        label = pred.numpy()  # label is an integer.

                    if output == 'logit':
                        return logits.numpy(), label
                    else:
                        return probas.numpy(), label

            self.predict_fn = predict_fn


class IntermediateLayerInterpreter(Interpreter):
    """This is one of the sub-abstract Interpreters. 
    
    :class:`IntermediateLayerInterpreter` exhibits features from intermediate layers to produce explanations.
    This interpreter extracts intermediate layers' features, but no gradients involved.
    Interpreters that are derived from :class:`IntermediateLayerInterpreter` include
    :class:`RolloutInterpreter`, :class:`ScoreCAMInterpreter`.

    This Interpreter implements :py:func:`_build_predict_fn` that returns the model's intermediate outputs given an 
    input. 
    """

    def __init__(self, paddle_model: callable, device: str, use_cuda: bool = None, **kwargs):
        """

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions.
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc.
        """

        Interpreter.__init__(self, paddle_model, device, use_cuda, **kwargs)
        assert hasattr(paddle_model, 'forward'), \
            "paddle_model has to be " \
            "an instance of paddle.nn.Layer or a compatible one."

    def _build_predict_fn(self, rebuild: bool = False, target_layer: str = None, target_layer_pattern: str = None):
        """Build :py:attr:`predict_fn` for IntermediateLayer based algorithms.
        The model is supposed to be a classification model.
        ``target_layer`` and ``target_layer_pattern`` cannot be set at the same time. See the arguments below.

        Args:
            rebuild (bool, optional): forces to rebuild. Defaults to ``False``.
            target_layer (str, optional): the name of the desired layer whose features will output. This is used when
                there is only one layer to output. Conflict with ``target_layer_pattern``. Defaults to ``None``.
            target_layer_pattern (str, optional): the pattern name of the layers whose features will output. This is 
                used when there are several layers to output and they share a common pattern name. Conflict with 
                ``target_layer``. Defaults to ``None``.
        """

        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        if self.predict_fn is None or rebuild:
            assert not (target_layer is None and target_layer_pattern is None), 'one of them must be given.'
            assert target_layer is None or target_layer_pattern is None, 'they cannot be given at the same time.'

            self._paddle_env_setup()

            def predict_fn(data):
                import paddle
                import re

                def target_layer_pattern_match(layer_name):
                    return re.match(target_layer_pattern, layer_name)

                def target_layer_match(layer_name):
                    return layer_name == target_layer

                match_func = target_layer_match if target_layer is not None else target_layer_pattern_match

                target_feature_maps = []

                def hook(layer, input, output):
                    target_feature_maps.append(output.numpy())

                hooks = []
                for name, v in self.paddle_model.named_sublayers():
                    if match_func(name):
                        h = v.register_forward_post_hook(hook)
                        hooks.append(h)

                assert len(hooks) > 0, f"No target layers are found in the given model, \
                                the list of layer names are \n \
                                {[n for n, v in self.paddle_model.named_sublayers()]}"

                with paddle.no_grad():
                    data = paddle.to_tensor(data)
                    logits = self.paddle_model(data)

                    # hooks has to be removed.
                    for h in hooks:
                        h.remove()

                    probas = paddle.nn.functional.softmax(logits, axis=1)
                    predict_label = paddle.argmax(probas, axis=1)  # get predictions.

                return target_feature_maps, probas.numpy(), predict_label.numpy()

            self.predict_fn = predict_fn
