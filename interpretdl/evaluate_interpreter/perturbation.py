import numpy as np
from .abc_evaluator import InterpreterEvaluator
from interpretdl.data_processor.readers import images_transform_pipeline, preprocess_image


class Perturbation(InterpreterEvaluator):
    """Perturbation based Evaluations. 

    More details of the Most Relevant First (MoRF) can be found in the original paper:
    https://arxiv.org/abs/1509.06321. 

    We additionally provide the Least Relevant First (LeRF) for the supplement. 
    Note that MoRF is equivalent to Deletion, but LeRF is NOT equivalent to Insertion.
    """

    def __init__(self, paddle_model: callable, device: str, compute_MoRF=True, compute_LeRF=True, **kwargs):
        super().__init__(paddle_model, device, None, **kwargs)

        if (not compute_MoRF) or (not compute_LeRF):
            raise ValueError('Either compute_MoRF or compute_LeRF is False.')
        self.compute_MoRF = compute_MoRF
        self.compute_LeRF = compute_LeRF
        self.evaluate_lime = False

        self._build_predict_fn()

    def _build_predict_fn(self, rebuild: bool=False):
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

                logits = self.paddle_model(paddle.to_tensor(data))  # get logits, [bs, num_c]
                probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
                return probas.numpy()

            self.predict_fn = predict_fn

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
                indices = np.random.choice(len(MoRF_images), limit_number_generated_samples)
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
                indices = np.random.choice(len(LeRF_images), limit_number_generated_samples)
                LeRF_images = [LeRF_images[i] for i in indices]

            LeRF_images = np.vstack(LeRF_images)
            results['LeRF_images'] = LeRF_images

        return results

    def generate_samples_array(self, img, explanation, limit_number_generated_samples, results):
        
        # usually explanation has shape of [n_sample, n_channel, h, w]
        if len(explanation.shape) == 4:
            assert explanation.shape[0] == 1, 'Explanation for one image.'
            explanation = explanation[0]
        if len(explanation.shape) == 3:
            explanation = np.abs(explanation).sum(0)
        assert len(explanation.shape) == 2

        if limit_number_generated_samples is None:
            limit_number_generated_samples = 20  # default to 20, each 5 percentiles.
        
        q = 100. / limit_number_generated_samples
        qs = [q*(i-1) for i in range(limit_number_generated_samples, 0, -1)]
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

    def compute_probas(self, results, coi=None):
        if self.compute_MoRF:
            data = preprocess_image(results['MoRF_images'])
            probas = self.predict_fn(data)

            # class of interest
            if coi is None:
                # probas.shape = [n_samples, n_classes]
                coi = np.argmax(probas[0], axis=0)
            
            results['MoRF_probas'] = probas[:, coi]
            results['MoRF_score'] = np.mean(results['MoRF_probas'])

        if self.compute_LeRF:
            data = preprocess_image(results['LeRF_images'])
            probas = self.predict_fn(data)

            # class of interest
            if coi is None:
                # probas.shape = [n_samples, n_classes]
                coi = np.argmax(probas[0], axis=0)
            
            results['LeRF_probas'] = probas[:, coi]
            results['LeRF_score'] = np.mean(results['LeRF_probas'])            
                
        return results

    def evaluate(self, img_path: str, explanation: list or np.ndarray, resize_to=224, crop_to=None, limit_number_generated_samples=None):
        """Main function of this class. Evaluate whether the explanation is trustworthy for the model.

        Args:
            img_path (str): a string for image path.
            explanation (listornp.ndarray): [description]
            resize_to (int, optional): [description]. Defaults to 224.
            crop_to ([type], optional): [description]. Defaults to None.
            limit_number_generated_samples ([type], optional): [description]. Defaults to None.

        Returns:
            [dict]: contains `MoRF_score`, `MoRF_probas`, `MoRF_images` if compute_MoRF; LeRF likewise. 
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
        results = self.compute_probas(results)

        return results