
# InterpretDL Examples and Tutorials

This directory contains jupyter notebooks for the usage examples and tutorials of InterpretDL.

## Examples

Examples are common usages of specific algorithms (both Interpreter and Evaluator). Both the code blocks and the final visualization results are provided in each example, to get a quick understanding about how to use InterpretDL.

Interpretation Algorithms:

| Methods                                                                                                                    | Representation          | Model Type             | Example           |
|----------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|-------------------|
| [LIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lime.py)                            | Input Features          | Model-Agnostic         | [link1](example_lime_cv.ipynb) \| [link2](example_lime_cv_ViT.ipynb) |
| [LIME with Prior](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lime_prior.py)           | Input Features          | Model-Agnostic         | [link](example_lime_gp_cv.ipynb) |
| [GLIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/glime.py)           | Input Features          | Model-Agnostic         | [link](LIME_Variants_part2.ipynb) |
| [NormLIME/FastNormLIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/_normlime_base.py) | Input Features          | Model-Agnostic         | [link1](example_normlime_cv.ipynb) \| [link2](example_normlime_nlp.ipynb) |
| [LRP](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lrp.py)                              | Input Features          | Differentiable         | [link](lrp_cv.ipynb) |
| [SmoothGrad](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/smooth_grad.py)               | Input Features          | Differentiable         | [link](example_smooth_grad_cv.ipynb) |
| [IntGrad](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/integrated_gradients.py)         | Input Features          | Differentiable         | [link](example_int_grad_cv.ipynb)  |
| [GradSHAP](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/gradient_shap.py)               | Input Features          | Differentiable         | [link](example_grad_shap_cv.ipynb) |
| [Occlusion](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/occlusion.py)                  | Input Features          | Model-Agnostic         | [link](example_occlusion_cv.ipynb) |
| [GradCAM/CAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/gradient_cam.py)             | Intermediate   Features | Specific: CNNs         | [link](example_grad_cam_cv.ipynb) |
| [ScoreCAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/score_cam.py)                   | Intermediate   Features | Specific: CNNs         | [link](example_score_cam_cv.ipynb) |
| [Rollout](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/rollout.py)                      | Intermediate   Features | Specific: Transformers | [link](example_rollout_cv_ViT.ipynb) |
| [TAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/transition_attention_maps.py)        | Intermediate   Features | Specific: Transformers | [link](example_tam_cv_ViT.ipynb) |
| [ForgettingEvents](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/forgetting_events.py)   | Dataset-Level           | Differentiable         | [link](example_forgetting_events_cv.ipynb) |
| [TIDY (Training Data Analyzer)](TIDY.ipynb)              | Dataset-Level           | Differentiable         | [link](TIDY.ipynb) |
| [Consensus](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/consensus.py)                  | Features                | Cross-Model            | [link](example_consensus_cv.ipynb)  |
| [Generic Attention](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/generic_attention.py)                  | Input Features                | Specific: Transformers            | [cv-link](example_bt_ga_cv_ViT.ipynb) \| [nlp-link](ga-bt-ernie-2.0-en-sst-2.ipynb) ([nblink](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/ga-bt-ernie-2.0-en-sst-2.ipynb))* \| [bi-modal-link](example_ga_bi-modal.ipynb)  ([nblink](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/example_ga_bi-modal.ipynb))* |
| [Bidirectional Transformer Explanation](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/consensus.py)                  | Input Features                | Specific: Transformers            | [cv-link](example_bt_ga_cv_ViT.ipynb) \| [nlp-link](ga-bt-ernie-2.0-en-sst-2.ipynb) ([nblink](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/ga-bt-ernie-2.0-en-sst-2.ipynb))*  |
| [BHDF](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/training_dynamics.py)                  | Dataset-Level                | Specific: Transformers            | [link](example_beyond_manually_designed_feature_cv.ipynb)  |

\* For text visualizations, NBViewer gives better and colorful rendering results. 

Trustworthiness Evaluation Algorithms:

| Method   Name      | Additional Notes                             | Example |
|--------------------|----------------------------------------------|---------|
| [Perturbation](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/perturbation.py)       | AUC, AP scores of MoRF, LeRF                     | [link](example_perturbation.ipynb)        |
| [Deletion&Insertion](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/deletion_insertion.py) | AUC, AP scores of Del, Ins                       | [link](example_del_ins.ipynb)     |


Interpretability Evaluation Algorithms:

| Method   Name      | Additional Notes                             | Example |
|--------------------|----------------------------------------------|---------|
| [PointGame](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/localization.py)          | Based on ground truth (bbox or segmentation) | [link](example_pointgame.ipynb)   |


## Tutorials

Tutorials can be found in [tutorials](https://github.com/PaddlePaddle/InterpretDL/tree/master/tutorials).
