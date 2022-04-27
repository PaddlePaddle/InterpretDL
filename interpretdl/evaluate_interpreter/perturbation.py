import numpy as np
import cv2
from .abc_evaluator import InterpreterEvaluator
from interpretdl.data_processor.readers import images_transform_pipeline, preprocess_image


class Perturbation(InterpreterEvaluator):
    """Perturbation based Evaluations. 

    The evaluation of interpretation algorithms follows the intuition that flipping the most salient pixels first 
    should lead to high performance decay. Perturbation-based examples can therefore be used for the trustworthiness
    evaluations of interpretation algorithms. 

    Two metrics are provided: most relevant first (MoRF) and least relevant first (LeRF).

    The MoRF metric is computed as follows. The perturbation starts from an original image, perturbs (zeros 
    out) the most important pixels in the input, and then computes the responses of the trained model. So that a 
    curve, with ratios of perturbed pixels as x-axis and probabilities as y-axis, can be obtained and the area under
    this curve is the MoRF score.

    The LeRF metric is similar, but the perturbation perturbs (zeros out) the least important pixels in the input and
    then computes the responses of the trained model. A similar curve can be obtained and the area under this curve is
    the LeRF score.

    Note that MoRF is equivalent to Deletion, but LeRF is NOT equivalent to Insertion.

    More details of MoRF and LeRF can be found in the original paper:
    https://arxiv.org/abs/1509.06321. 

    """

    def __init__(self,
                 paddle_model: callable,
                 device: str,
                 compute_MoRF: bool = True,
                 compute_LeRF: bool = True,
                 **kwargs):
        """_summary_

        Args:
            paddle_model (callable): A model with :py:func:`forward` and possibly :py:func:`backward` functions. This 
                is not always required if the model is not involved. 
            device (str): The device used for running ``paddle_model``, options: ``"cpu"``, ``"gpu:0"``, ``"gpu:1"`` 
                etc. Again, this is not always required if the model is not involved.
            compute_MoRF (bool, optional): Whether comptue MoRF score. Defaults to True.
            compute_LeRF (bool, optional): Whether comptue LeRF score. Defaults to True.

        Raises:
            ValueError: 'At least one of ``compute_MoRF`` and ``compute_LeRF`` must be True.'
        """
        super().__init__(paddle_model, device, None, **kwargs)

        if (not compute_MoRF) and (not compute_LeRF):
            raise ValueError('At least one of ``compute_MoRF`` and ``compute_LeRF`` must be True.')
        self.compute_MoRF = compute_MoRF
        self.compute_LeRF = compute_LeRF
        self.evaluate_lime = False

        self._build_predict_fn()

    def _build_predict_fn(self, rebuild: bool = False):
        if self.predict_fn is not None:
            assert callable(self.predict_fn), "predict_fn is predefined before, but is not callable." \
                "Check it again."

        import paddle
        if self.predict_fn is None or rebuild:
            if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
                print("Paddle is not installed with GPU support. Change to CPU version now.")
                self.device = 'cpu'

            # set device. self.device is one of ['cpu', 'gpu:0', 'gpu:1', ...]
            paddle.set_device(self.device)

            # to get gradients, the ``train`` mode must be set.
            self.paddle_model.eval()

            def predict_fn(data):
                """predict_fn for input gradients based interpreters,
                    for image classification models only.

                Args:
                    data ([type]): [description]

                Returns:
                    [type]: [description]
                """
                assert len(data.shape) == 4  # [bs, h, w, 3]

                with paddle.no_grad():
                    logits = self.paddle_model(paddle.to_tensor(data))  # get logits, [bs, num_c]
                    probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
                return probas.numpy()

            self.predict_fn = predict_fn

    def evaluate(self,
                 img_path: str,
                 explanation: list or np.ndarray,
                 batch_size=None,
                 resize_to=224,
                 crop_to=None,
                 limit_number_generated_samples=None) -> dict:
        """Given ``img_path``, Perturbation first generates perturbed samples of MoRF and LeRF respectively, according
        to the order provided by ``explanation``. The number of samples is defined by 
        ``limit_number_generated_samples`` (a sampling is used for the numbers are different). Then Perturbation 
        computes the probabilities of these perturbed samples, and the mean of all probabilities of the class of 
        interest is computed for the final score.

        Note that LIME produces explanations based on superpixels, the number of perturbed samples is originally equal
        to the number of superpixels. So if ``limit_number_generated_samples`` is None, then the number of superpixels
        is used. For other explanations that produce the explanation of the same spatial dimension as the input image,
        ``limit_number_generated_samples`` is set to ``20`` if not given.

        Args:
            img_path (str): a string for image path.
            explanation (list or np.ndarray): the explanation result from an interpretation algorithm.
            batch_size (int or None, optional): batch size for each pass. Defaults to ``None``.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.
            limit_number_generated_samples (int or None, optional): a maximum value for samples of perturbation. If 
                None, it will be automatically chosen. The number of superpixels is used for LIME explanations, 
                otherwise, ``20`` is to be set. Defaults to ``None``.

        Returns:
            dict: 
                A dict containing ``'MoRF_score'``, ``'MoRF_probas'``, ``'MoRF_images'``, ``'LeRF_score'``, 
                ``'LeRF_probas'`` and ``'LeRF_images'``, if ``compute_MoRF`` and ``compute_LeRF`` are both True.
        """

        if not isinstance(explanation, np.ndarray):
            # if not an array, then should be lime results.
            # for lime results, superpixel segmentation corresponding to the lime_weights is required.
            assert isinstance(explanation, dict) and 'segmentation' in explanation, \
                'For LIME results, give the LIMECVInterpreter.lime_results as explanation. ' \
                'If there are confusions, please contact us.'
            self.evaluate_lime = True

        results = {}
        if self.compute_MoRF:
            results['MoRF_score'] = 0.0
            results['MoRF_probas'] = None
        if self.compute_LeRF:
            results['LeRF_score'] = 0.0
            results['LeRF_probas'] = None

        img, _ = images_transform_pipeline(img_path, resize_to=resize_to, crop_to=crop_to)
        results = self.generate_samples(img, explanation, limit_number_generated_samples, results)
        results = self.compute_probas(results, batch_size)

        return results

    def generate_samples(self, img, explanation, limit_number_generated_samples, results):

        if self.evaluate_lime:
            return self.generate_samples_lime(img, explanation, limit_number_generated_samples, results)
        else:
            return self.generate_samples_array(img, explanation, limit_number_generated_samples, results)

    def generate_samples_lime(self, img, explanation, limit_number_generated_samples, results):
        sp_segments = explanation['segmentation']
        lime_weights = explanation['lime_weights']
        interpret_class = list(lime_weights.keys())[0]

        sp_order = [sp for sp, v in lime_weights[interpret_class]]
        mx = (127, 127, 127)

        if self.compute_MoRF:
            MoRF_images = [img]
            fudged_image = img.copy()
            for sp in sp_order:
                fudged_image = fudged_image.copy()

                indices = np.where(sp_segments == sp)
                fudged_image[:, indices[0], indices[1]] = mx

                MoRF_images.append(fudged_image)

            if limit_number_generated_samples is not None and limit_number_generated_samples < len(MoRF_images):
                indices = np.linspace(0, len(MoRF_images) - 1, limit_number_generated_samples).astype(np.int32)
                MoRF_images = [MoRF_images[i] for i in indices]

            MoRF_images = np.vstack(MoRF_images)
            results['MoRF_images'] = MoRF_images

        if self.compute_LeRF:
            LeRF_images = [img]
            fudged_image = img.copy()
            for sp in reversed(sp_order):
                fudged_image = fudged_image.copy()

                indices = np.where(sp_segments == sp)
                fudged_image[:, indices[0], indices[1]] = mx

                LeRF_images.append(fudged_image)

            if limit_number_generated_samples is not None and limit_number_generated_samples < len(LeRF_images):
                indices = np.linspace(0, len(LeRF_images) - 1, limit_number_generated_samples).astype(np.int32)
                LeRF_images = [LeRF_images[i] for i in indices]

            LeRF_images = np.vstack(LeRF_images)
            results['LeRF_images'] = LeRF_images

        return results

    def generate_samples_array(self, img, explanation, limit_number_generated_samples, results):

        # usually explanation has shape of [n_sample, n_channel, h, w]
        explanation = np.squeeze(explanation)
        assert len(explanation.shape) in [2, 3], 'Explanation for one image.'
        if len(explanation.shape) == 3:
            explanation = np.abs(explanation).sum(0)
        assert len(explanation.shape) == 2

        explanation = cv2.resize(explanation, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_LINEAR)

        if limit_number_generated_samples is None:
            limit_number_generated_samples = 20  # default to 20, each 5 percentiles.

        q = 100. / limit_number_generated_samples
        qs = [q * (i - 1) for i in range(limit_number_generated_samples, 0, -1)]
        percentiles = np.percentile(explanation, qs)
        mx = (127, 127, 127)

        if self.compute_MoRF:
            MoRF_images = [img]
            fudged_image = img.copy()
            for p in percentiles:
                fudged_image = fudged_image.copy()
                indices = np.where(explanation > p)
                fudged_image[:, indices[0], indices[1]] = mx
                MoRF_images.append(fudged_image)
            MoRF_images = np.vstack(MoRF_images)
            results['MoRF_images'] = MoRF_images

        if self.compute_LeRF:
            LeRF_images = [img]
            fudged_image = img.copy()
            for p in percentiles[::-1]:
                fudged_image = fudged_image.copy()
                indices = np.where(explanation < p)
                fudged_image[:, indices[0], indices[1]] = mx
                LeRF_images.append(fudged_image)
            LeRF_images = np.vstack(LeRF_images)
            results['LeRF_images'] = LeRF_images

        return results

    def compute_probas(self, results, batch_size, coi=None):
        if self.compute_MoRF:
            data = preprocess_image(results['MoRF_images'])
            if batch_size is None:
                probas = self.predict_fn(data)
            else:
                probas = []
                list_to_compute = list(range(data.shape[0]))
                while len(list_to_compute) > 0:
                    if len(list_to_compute) >= batch_size:
                        list_c = list_to_compute[:batch_size]
                        list_to_compute = list_to_compute[batch_size:]
                    else:
                        list_c = list_to_compute[:len(list_to_compute)]
                        list_to_compute = []

                    probs_batch = self.predict_fn(data[list_c])
                    probas.append(probs_batch)

                probas = np.concatenate(probas, axis=0)

            # class of interest
            if coi is None:
                # probas.shape = [n_samples, n_classes]
                coi = np.argmax(probas[0], axis=0)

            results['MoRF_probas'] = probas[:, coi]
            results['MoRF_score'] = np.mean(results['MoRF_probas'])

        if self.compute_LeRF:
            data = preprocess_image(results['LeRF_images'])
            if batch_size is None:
                probas = self.predict_fn(data)
            else:
                probas = []
                list_to_compute = list(range(data.shape[0]))
                while len(list_to_compute) > 0:
                    if len(list_to_compute) >= batch_size:
                        list_c = list_to_compute[:batch_size]
                        list_to_compute = list_to_compute[batch_size:]
                    else:
                        list_c = list_to_compute[:len(list_to_compute)]
                        list_to_compute = []

                    probs_batch = self.predict_fn(data[list_c])
                    probas.append(probs_batch)

                probas = np.concatenate(probas, axis=0)

            # class of interest
            if coi is None:
                # probas.shape = [n_samples, n_classes]
                coi = np.argmax(probas[0], axis=0)

            results['LeRF_probas'] = probas[:, coi]
            results['LeRF_score'] = np.mean(results['LeRF_probas'])

        return results
