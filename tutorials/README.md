
# InterpretDL Examples and Tutorials

This directory contains jupyter notebooks for the usage examples and tutorials of InterpretDL.

## Examples

Examples are common usages of specific algorithms (both Interpreter and Evaluator). Both the code blocks and the final visualization results are provided in each example, to get a quick understanding about how to use InterpretDL.

Interpretation Algorithms:

| Methods                                                                                                                    | Representation          | Model Type             | Example           |
|----------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|-------------------|
| [LIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lime.py)                            | Input Features          | Model-Agnostic         | [link1](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_lime_cv.ipynb) \| [link2](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_lime_cv_ViT.ipynb) |
| [LIME with Prior](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lime_prior.py)           | Input Features          | Model-Agnostic         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_lime_gp_cv.ipynb) |
| [GLIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/glime.py)           | Input Features          | Model-Agnostic         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/LIME_Variants_part2.ipynb) |
| [NormLIME/FastNormLIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/_normlime_base.py) | Input Features          | Model-Agnostic         | [link1](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_normlime_cv.ipynb) \| [link2](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_normlime_nlp.ipynb) |
| [LRP](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lrp.py)                              | Input Features          | Differentiable         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/lrp_cv.ipynb) |
| [SmoothGrad](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/smooth_grad.py)               | Input Features          | Differentiable         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_smooth_grad_cv.ipynb) |
| [IntGrad](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/integrated_gradients.py)         | Input Features          | Differentiable         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_int_grad_cv.ipynb)  |
| [GradSHAP](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/gradient_shap.py)               | Input Features          | Differentiable         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_grad_shap_cv.ipynb) |
| [Occlusion](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/occlusion.py)                  | Input Features          | Model-Agnostic         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_occlusion_cv.ipynb) |
| [GradCAM/CAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/gradient_cam.py)             | Intermediate   Features | Specific: CNNs         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_grad_cam_cv.ipynb) |
| [ScoreCAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/score_cam.py)                   | Intermediate   Features | Specific: CNNs         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_score_cam_cv.ipynb) |
| [Rollout](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/rollout.py)                      | Intermediate   Features | Specific: Transformers | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_rollout_cv_ViT.ipynb) |
| [TAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/transition_attention_maps.py)        | Intermediate   Features | Specific: Transformers | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_tam_cv_ViT.ipynb) |
| [ForgettingEvents](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/forgetting_events.py)   | Dataset-Level           | Differentiable         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_forgetting_events_cv.ipynb) |
| [TIDY (Training Data Analyzer)](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/TIDY.ipynb)              | Dataset-Level           | Differentiable         | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/TIDY.ipynb) |
| [Consensus](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/consensus.py)                  | Features                | Cross-Model            | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_consensus_cv.ipynb)  |
| [Generic Attention](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/generic_attention.py)                  | Input Features                | Specific: Bi-Modal Transformers            | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_ga_bi-modal.ipynb)  ([nblink](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/example_ga_bi-modal.ipynb))*|

\* For text visualizations, NBViewer gives better and colorful rendering results. 

Trustworthiness Evaluation Algorithms:

| Method   Name      | Additional Notes                             | Example |
|--------------------|----------------------------------------------|---------|
| [Perturbation](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/perturbation.py)       | AUC, AP scores of MoRF, LeRF                     | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_perturbation.ipynb)        |
| [Deletion&Insertion](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/deletion_insertion.py) | AUC, AP scores of Del, Ins                       | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_del_ins.ipynb)     |
| [PointGame](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/localization.py)          | Based on ground truth (bbox or segmentation) | [link](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_pointgame.ipynb)   |



## Tutorials

Usage examples do not discuss the details of these algorithms. Therefore, for both practical and academic purposes, we are preparing a series of tutorials to introduce the designs and motivations of interpreters and trustworthiness evaluators.

The available (and planning) tutorials are listed below:

- [Getting Started](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/Getting_Started.ipynb). This tutorial includes the installation and basic ideas of InterpretDL.

- NLP Explanations. There are four tutorials for NLP tutorials, using 
[Ernie2.0 in English](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb)), 
[Bert in English](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/bert-en-sst-2.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/bert-en-sst-2.ipynb)), 
[BiLSTM in Chinese](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/bilstm-zh-chnsenticorp.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/bilstm-zh-chnsenticorp.ipynb)) and 
[Ernie1.0 in Chinese](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-1.0-zh-chnsenticorp.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-1.0-zh-chnsenticorp.ipynb))
as examples. For text visualizations, NBViewer gives better and colorful rendering results.

- [Input Gradient Interpreters](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/Input_Gradient.ipynb). This tutorial introduces the input gradient based interpretation algorithms.

- LIME and Its Variants [Part1](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/LIME_Variants_part1.ipynb) (LIME) | [Part2](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/LIME_Variants_part2.ipynb) (GLIME). This tutorial introduces the LIME algorithms and many advanced improvements based on LIME.

- Transformers (to appear).

- Trustworthiness Evaluation Tutorials (to appear).

- Dataset-Level Tutorials (to appear).