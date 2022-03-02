import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class PointGame():
    """Pointing Game Evaluation Method.

    More details can be found in the original paper:
    https://arxiv.org/abs/1608.00507.

    Note that the bounding box of annotations is required for the evaluation.
    This method does not need models either.
    """
    def __init__(self):
        pass

    def evaluate(self, bbox, exp_array, threshold=0.25):
        ret = np.max(exp_array) * threshold
        binary_exp_array = exp_array > ret
        
        gt = np.zeros_like(exp_array, dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        gt[y1:y2, x1:x2] = 1
        
        TP = (binary_exp_array * gt).sum()
        predict_pos = binary_exp_array.sum()
        actual_pos = gt.sum()
        
        precision = TP / predict_pos
        recall = TP / actual_pos
        f1_score = (2*precision*recall) / (precision + recall + 1e-6)

        # depends on the threshold
        r = {'precision': precision, 'recall': recall, 'f1_score': f1_score}

        # independ of threshold
        auc_score = roc_auc_score(gt.flatten(), exp_array.flatten())
        ap_score = average_precision_score(gt.flatten(), exp_array.flatten())
        r.update( {'auc_score': auc_score, 'ap_score': ap_score} )

        return r