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
        **Should not be called explicitly**.

        Args:
            predict_fn: A defined callable function that defines inputs and outputs.
                Defaults to None, and each interpreter will generate it.
                example for LIME:
                    def get_predict_fn():
                        startup_prog = fluid.Program()
                        main_program = fluid.Program()
                        with fluid.program_guard(main_program, startup_prog):
                            with fluid.unique_name.guard():
                                image_op = fluid.data(
                                    name='image',
                                    shape=[None] + model_input_shape,
                                    dtype='float32')
                                # paddle model
                                class_num = 1000
                                model = ResNet101()
                                logits = model.net(input=image_input, class_dim=class_num)
                                probs = fluid.layers.softmax(logits, axis=-1)
                                if isinstance(probs, tuple):
                                    probs = probs[0]
                                # end of paddle model
                                main_program = main_program.clone(for_test=True)

                        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
                        place = fluid.CUDAPlace(gpu_id)
                        exe = fluid.Executor(place)

                        fluid.io.load_persistables(exe, trained_model_path,
                                                   main_program)

                        def predict_fn(visual_images):
                            images = preprocess_image(
                                visual_images
                            )  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
                            [result] = exe.run(main_program,
                                               fetch_list=[probs],
                                               feed={'image': images})

                            return result

                        return predict_fn

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
