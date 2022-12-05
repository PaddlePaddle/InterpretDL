import cv2
import numpy as np
from .abc_evaluator import InterpreterEvaluator
from ..data_processor.readers import images_transform_pipeline, preprocess_image


class Infidelity(InterpreterEvaluator):
    """
    Infidelity Interpreter Evaluation method.

    The idea of fidelity is similar to the faithfulness evaluation, to evaluate how faithful/reliable/loyal of the
    explanations to the model. (In)fidelity measures the normalized squared Euclidean distance between two terms:
    the product of a perturbation and the explanation, and the difference between the model's response to the original
    input and the one to the perturbed input, i.e.

    .. math::
        INFD(\Phi, f, x) = \mathbb{E}_{I \sim \mu_I} [ (I^T \Phi(f, x) - (f(x) - f(x - I)) )^2 ],

    where the meaning of the symbols can be found in the original paper.

    A normalization is added, which is not in the paper but in the 
    `official implementation <https://github.com/chihkuanyeh/saliency_evaluation>`_: 
    
    .. math::
        \\beta = \\frac{
            \mathbb{E}_{I \sim \mu_I} [ I^T \Phi(f, x) (f(x) - f(x - I)) ]
            }{
                \mathbb{E}_{I \sim \mu_I} [ (I^T \Phi(f, x))^2 ]
            }

    Intuitively, given a perturbation, e.g., a perturbation on important pixels, the product (the former term) should 
    be relatively large if the explanation indicates the important pixels too, compared to a perturbation on irrelavant
    pixels; while the difference (the latter term) should also be large because the model depends on important pixels
    to make decisions. Like this, large values would be offset by large values if the explanation is faithful to the 
    model. Otherwise, for uniform explanations (all being constant), the former term would be a constant value and the
    infidelity would become large.

    More details about the measure can be found in the original paper: https://arxiv.org/abs/1901.09392.
    """
    def __init__(self,
                model: callable,
                device: str = 'gpu:0',
                **kwargs):
        """

        Args:
            model (callable): _description_
            device (_type_, optional): _description_. Defaults to 'gpu:0'.
        """
        
        super().__init__(model, device, **kwargs)
        self.results = {}

    def _build_predict_fn(self, rebuild: bool = False):
        """Different from InterpreterEvaluator._build_predict_fn(): using logits.

        Args:
            rebuild (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
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
            self.model.eval()

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
                    # Follow the `official implementation <https://github.com/chihkuanyeh/saliency_evaluation>`_
                    # to use logits as output.
                    logits = self.model(paddle.to_tensor(data))  # get logits, [bs, num_c]
                    # probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
                return logits.numpy()

            self.predict_fn = predict_fn
    
    def _generate_samples(self, img):
        copy_data = np.copy(img)
        bs, height, width, color_channel = copy_data.shape

        generated_samples = []
        Is = []

        # fixed kernel_size. independent of explanations.
        kernel_size = [32, 64, 128]
        # Note that `official implementation <https://github.com/chihkuanyeh/saliency_evaluation>`_ uses the stride of
        # 1. Here we use a large stride of 8 for reducing computations.
        stride = 8

        for k in kernel_size:
            h_range = ( height - stride ) // stride
            w_range = ( width - stride ) // stride
            if h_range * stride < height:
                h_range += 1
            if w_range * stride < width:
                w_range += 1

            for i in range(h_range):
                start_h = i * stride
                end_h = start_h + k
                if end_h > height:
                    end_h = height
                    break
                for j in range(w_range):
                    start_w = j * stride
                    end_w = start_w + k
                    if end_w > width:
                        end_w = width
                        break
                    tmp_data = np.copy(img)
                    tmp_data[:, start_h:end_h, start_w:end_w, :] = 127
                    
                    Is.append(img != tmp_data)  # binary I.
                    generated_samples.append(tmp_data)

        # print(len(generated_samples))

        return np.concatenate(generated_samples, axis=0), \
            np.concatenate(Is, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
    
    def evaluate(self, 
                 img_path: str or np.ndarray, 
                 explanation: np.ndarray, 
                 recompute: bool = False, 
                 batch_size: int = 50, 
                 resize_to: int = 224, 
                 crop_to: None or int = None):
        """Given ``img_path``, Infidelity first generates perturbed samples, with a square removal strategy on the 
        original image. Since the difference (the second term in the infidelity formula) is independ of the 
        explanation, so we temporaily save these results in case this image has other explanations for evaluations.

        Then, given ``explanation``, we follow the formula to compute the infidelity. A normalization is added, 
        which is not in the paper but in the 
        `official implementation <https://github.com/chihkuanyeh/saliency_evaluation>`_.

        Args:
            img_path (strornp.ndarray): a string for image path.
            explanation (np.ndarray): the explanation result from an interpretation algorithm.
            recompute (bool, optional): whether forcing to recompute. Defaults to False.
            batch_size (int, optional): batch size for each pass.. Defaults to 50.
            resize_to (int, optional): Images will be rescaled with the shorter edge being ``resize_to``. Defaults to 
                ``224``.
            crop_to (int, optional): After resize, images will be center cropped to a square image with the size 
                ``crop_to``. If None, no crop will be performed. Defaults to ``None``.

        Returns:
            int: the infidelity score.
        """
        self._build_predict_fn()

        explanation = explanation.squeeze()
        assert len(explanation.shape) == 2, \
            f"Explanation should only have two dimensions after squeezed but got shape of {explanation.shape}."

        img, data = images_transform_pipeline(img_path, resize_to=resize_to, crop_to=crop_to)

        if 'proba_diff' not in self.results or recompute:
            ## x and I related.
            generated_samples, Is = self._generate_samples(img)
            self.results['generated_samples'] = generated_samples
            self.results['Is'] = Is

            generated_samples = preprocess_image(generated_samples)
            probas_x = self.predict_fn(data)
            label = np.argmax(probas_x[0])
            proba_x = probas_x[:, label]
            self.results['predict'] = {'label': label, 'proba': proba_x}

            if batch_size is None:
                proba_pert = self.predict_fn(generated_samples)[:, label]
            else:
                proba_pert = []
                list_to_compute = list(range(generated_samples.shape[0]))
                while len(list_to_compute) > 0:
                    if len(list_to_compute) >= batch_size:
                        list_c = list_to_compute[:batch_size]
                        list_to_compute = list_to_compute[batch_size:]
                    else:
                        list_c = list_to_compute[:len(list_to_compute)]
                        list_to_compute = []

                    probs_batch = self.predict_fn(generated_samples[list_c])[:, label]
                    proba_pert.append(probs_batch)
                proba_pert = np.concatenate(proba_pert)
            # proba_pert = self.predict_fn(generated_samples)[:, label]
            proba_diff = proba_x - proba_pert

            self.results['proba_diff'] = proba_diff
        else:
            Is = self.results['Is']
            proba_diff = self.results['proba_diff']

        ## explanation related.
        resized_exp = cv2.resize(explanation, (data.shape[2], data.shape[3]))
        resized_exp = resized_exp.reshape((1, 1, data.shape[2], data.shape[3]))
        exp_sum = np.sum(Is * resized_exp, axis=(1, 2, 3))

        # performs optimal scaling for each explanation before calculating the infidelity score
        if np.mean(exp_sum*exp_sum) == 0.0:
            exp_sum = 0.0  # simple handling the NAN issue.
        else:
            beta = (proba_diff*exp_sum).mean() / np.mean(exp_sum*exp_sum)
            exp_sum *= beta

        infid = np.mean(np.square(proba_diff-exp_sum))

        self.results['explanation'] = explanation
        self.results['infid'] = infid

        return infid


class InfidelityNLP(InterpreterEvaluator):
    def __init__(self, model: callable or None, device: str = 'gpu:0', **kwargs):
        super().__init__(model, device, **kwargs)
        self.results = {}

    def _generate_samples(self, input_ids, masked_id: int, is_random_samples: bool):
        num_tokens = len(input_ids)

        if is_random_samples:
            # This is more suitable for long documents.
            # we concat three kinds of perturbations: 
            # randomly perturbing 1%, 2%, 3%, 4% or 5% tokens respectively
            # with 40 times
            num_repeats = 40
            results = []
            ids_array = np.array([input_ids]*num_repeats)
            for p in range(1, 6):
                _k = int(num_tokens * p / 100)

                # not choose from {0, -1}, i.e., [CLS] and [SEP]
                # https://stackoverflow.com/a/53893160/4834515
                pert_k = np.random.rand(num_repeats, num_tokens-2).argpartition(_k, axis=1)[:,:_k] + 1

                pert_array = np.copy(ids_array)
                # vectorized slicing.
                # https://stackoverflow.com/a/74024396/4834515
                row_indexes = np.arange(num_repeats)[:, None]
                pert_array[row_indexes, pert_k] = masked_id

                results.append(pert_array)

            perturbed_samples = np.concatenate(results)  # [200, num_tokens]
            Is = perturbed_samples != np.array([input_ids])  # [200, num_tokens]

            return perturbed_samples, Is
        else:
            # This is more suitable for short documents.
            # like 1d-conv, stride=1, kernel-size={1,2,3,4,5}
            generated_samples = []
            input_ids_array = np.array([input_ids])
            for ks in range(1, 6):
                if ks > num_tokens - 2:
                    break
                for i in range(1, num_tokens-ks):
                    tmp = np.copy(input_ids_array)
                    tmp[0, i:i+ks] = masked_id
                    generated_samples.append(tmp)
            
            perturbed_samples = np.concatenate(generated_samples, axis=0)
            Is = perturbed_samples != input_ids_array

            return perturbed_samples, Is

    # def _generate_samples(self, input_ids, masked_id=0):
    #     num_tokens = len(input_ids)

    #     # we concat three kinds of perturbations: 
    #     # randomly perturbing 1, 2 or 3 tokens respectively
    #     # with 33 times
    #     num_repeats = 33

    #     ids_array = np.array([input_ids]*num_repeats)

    #     # not choose from {0, -1}, [CLS] and [SEP]
    #     # https://stackoverflow.com/a/53893160/4834515
    #     pert_1 = np.random.rand(num_repeats, num_tokens-2).argpartition(1, axis=1)[:,:1] + 1
    #     pert_2 = np.random.rand(num_repeats, num_tokens-2).argpartition(2, axis=1)[:,:2] + 1
    #     pert_3 = np.random.rand(num_repeats, num_tokens-2).argpartition(3, axis=1)[:,:3] + 1

    #     pert_1_array = np.copy(ids_array)
    #     pert_2_array = np.copy(ids_array)
    #     pert_3_array = np.copy(ids_array)

    #     # https://stackoverflow.com/a/74024396/4834515
    #     row_indexes = np.arange(num_repeats)[:, None]
    #     pert_1_array[row_indexes, pert_1] = masked_id
    #     pert_2_array[row_indexes, pert_2] = masked_id
    #     pert_3_array[row_indexes, pert_3] = masked_id

    #     perturbed_samples = np.concatenate([pert_1_array, pert_2_array, pert_3_array])
    #     return perturbed_samples, perturbed_samples != ids_array

    def evaluate(self, raw_text: str, explanation: list or np.ndarray, tokenizer: callable, max_seq_len=128, is_random_samples=False, recompute: bool = False):
        self._build_predict_fn()

        # tokenizer text to ids
        encoded_inputs = tokenizer(raw_text, max_seq_len=max_seq_len)
        # order is important. *_batched_and_to_tuple will be the input for the model.
        _batched_and_to_tuple = tuple([np.array([v]) for v in encoded_inputs.values()])

        probas_x = self.predict_fn(_batched_and_to_tuple)
        label = np.argmax(probas_x[0])
        proba_x = probas_x[:, label]
        self.results['predict'] = {'label': label, 'proba': proba_x}

        explanation = np.squeeze(explanation)
        assert explanation.shape[0] == len(encoded_inputs['input_ids'])
        
        # generate perturbation samples.
        if 'proba_diff' not in self.results or recompute:
            ## x and I related.
            generated_samples, Is = self._generate_samples(encoded_inputs['input_ids'], tokenizer.pad_token_id, is_random_samples)
            self.results['generated_samples'] = generated_samples
            self.results['Is'] = Is
            proba_pert = self.predict_fn(generated_samples)[:, label]
            proba_diff = proba_x - proba_pert
            self.results['proba_diff'] = proba_diff
        else:
            Is = self.results['Is']
            proba_diff = self.results['proba_diff']

        ## explanation related.
        exp_sum = np.sum(Is * explanation, axis=(-1))

        # performs optimal scaling for each explanation before calculating the infidelity score
        if np.mean(exp_sum*exp_sum) == 0.0:
            exp_sum = 0.0  # simple handling the NAN issue.
        else:
            beta = (proba_diff*exp_sum).mean() / np.mean(exp_sum*exp_sum)
            exp_sum *= beta

        infid = np.mean(np.square(proba_diff-exp_sum))

        self.results['explanation'] = explanation
        self.results['infid'] = infid

        return infid
