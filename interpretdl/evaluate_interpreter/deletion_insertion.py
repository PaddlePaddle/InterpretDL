import numpy as np
from .abc_evaluator import InterpreterEvaluator
from interpretdl.data_processor.readers import images_transform_pipeline, preprocess_image

class DeletionInsertion(InterpreterEvaluator):
    """
    Deletion & Insertion Interpreter Evaluation method.

    More details regarding the Deletion & Insertion method can be found in the original paper:
    https://arxiv.org/abs/1806.07421
    """
    
    def __init__(self, paddle_model: callable, device: str, use_cuda: bool, compute_deletion=True, compute_insertion=True, **kwargs):
        super().__init__(paddle_model, device, use_cuda, **kwargs)

        if (not compute_deletion) or (not compute_insertion):
            raise ValueError('Either compute_deletion or compute_insertion is False.')
        self.compute_deletion = compute_deletion
        self.compute_insertion = compute_insertion
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

        if self.compute_deletion:
            deletion_images = [img]
            sp_order = [sp for sp, v in lime_weights[interpret_class]]
            fudged_image = img.copy()
            for sp in sp_order:
                fudged_image = fudged_image.copy()
                mx = (127, 127, 127)

                indices = np.where(sp_segments == sp)
                fudged_image[:, indices[0], indices[1]] = mx
                
                deletion_images.append(fudged_image)

            if limit_number_generated_samples is not None and limit_number_generated_samples < len(deletion_images):
                indices = np.random.choice(len(deletion_images), limit_number_generated_samples)
                deletion_images = [deletion_images[i] for i in indices]
            
            deletion_images = np.vstack(deletion_images)
            results['deletion_images'] = deletion_images
        
        if self.compute_insertion:
            insertion_images = []
            fudged_image = np.zeros_like(img) + 127
            for sp in sp_order:
                fudged_image = fudged_image.copy()

                indices = np.where(sp_segments == sp)
                fudged_image[:, indices[0], indices[1]] = img[:, indices[0], indices[1]]
                
                insertion_images.append(fudged_image)
                
            insertion_images.append(img)

            if limit_number_generated_samples is not None and limit_number_generated_samples < len(insertion_images):
                indices = np.random.choice(len(insertion_images), limit_number_generated_samples)
                insertion_images = [insertion_images[i] for i in indices]

            insertion_images = np.vstack(insertion_images)
            results['insertion_images'] = insertion_images

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
        
        if self.compute_deletion:
            deletion_images = [img]

            fudged_image = img.copy()
            for p in percentiles:
                fudged_image = fudged_image.copy()
                mx = (127, 127, 127)
                indices = np.where(explanation > p)
                fudged_image[:, indices[0], indices[1]] = mx
                deletion_images.append(fudged_image)
            deletion_images = np.vstack(deletion_images)
            results['deletion_images'] = deletion_images

        if self.compute_insertion:
            insertion_images = []
            fudged_image = np.zeros_like(img) + 127
            for p in percentiles:
                fudged_image = fudged_image.copy()
                indices = np.where(explanation > p)
                fudged_image[:, indices[0], indices[1]] = img[:, indices[0], indices[1]]
                insertion_images.append(fudged_image)
            insertion_images.append(img)

            insertion_images = np.vstack(insertion_images)
            results['insertion_images'] = insertion_images            
        
        return results

    def compute_probas(self, results, coi=None):
        if self.compute_deletion:
            data = preprocess_image(results['deletion_images'])
            probas = self.predict_fn(data)

            # class of interest
            if coi is None:
                # probas.shape = [n_samples, n_classes]
                coi = np.argmax(probas[0], axis=0)
            
            results['del_probas'] = probas[:, coi]
            results['deletion_score'] = np.mean(results['del_probas'])

        if self.compute_insertion:
            data = preprocess_image(results['insertion_images'])
            probas = self.predict_fn(data)

            # class of interest
            if coi is None:
                # probas.shape = [n_samples, n_classes]
                coi = np.argmax(probas[-1], axis=0)
            
            results['ins_probas'] = probas[:, coi]
            results['insertion_score'] = np.mean(results['ins_probas'])            
                
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
            [dict]: contains `deletion_score`, `del_probas`, `deletion_images` if compute_deletion; insertion likewise. 
        """
        
        if not isinstance(explanation, np.ndarray):
            # if not an array, then should be lime results.
            # for lime results, superpixel segmentation corresponding to the lime_weights is required.
            assert isinstance(explanation, dict) and 'segmentation' in explanation, \
                'For LIME results, give the LIMECVInterpreter.lime_results as explanation. ' \
                'If there are confusions, please contact us.'
            self.evaluate_lime = True

        results = {}
        if self.compute_deletion:
            results['deletion_score'] = 0.0
            results['del_probas'] = None
        if self.compute_insertion:
            results['insertion_score'] = 0.0
            results['ins_probas'] = None

        img, _ = images_transform_pipeline(img_path, resize_to=resize_to, crop_to=crop_to)
        results = self.generate_samples(img, explanation, limit_number_generated_samples, results)
        results = self.compute_probas(results)

        return results