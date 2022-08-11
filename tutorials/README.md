
# InterpretDL Examples and Tutorials

This directory contains jupyter notebooks for tutorials of InterpretDL.

## Usage Examples
Usages examples are moved to [examples](https://github.com/PaddlePaddle/InterpretDL/tree/master/examples).

## Tutorials

Usage examples do not discuss the details of these algorithms. Therefore, for both practical and academic purposes, we are preparing a series of tutorials to introduce the designs and motivations of interpreters and trustworthiness evaluators.

The available (and planning) tutorials are listed below:

- [Getting Started](Getting_Started.ipynb). This tutorial includes the installation and basic ideas of InterpretDL.

- NLP Explanations. There are four tutorials for NLP tutorials, using 
[Ernie2.0 in English](ernie-2.0-en-sst-2.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb)), 
[Bert in English](bert-en-sst-2.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/bert-en-sst-2.ipynb)), 
[BiLSTM in Chinese](bilstm-zh-chnsenticorp.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/bilstm-zh-chnsenticorp.ipynb)) and 
[Ernie1.0 in Chinese](ernie-1.0-zh-chnsenticorp.ipynb) ([on NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-1.0-zh-chnsenticorp.ipynb))
as examples. For text visualizations, NBViewer gives better and colorful rendering results.

- [Input Gradient Interpreters](Input_Gradient.ipynb). This tutorial introduces the input gradient based interpretation algorithms.

- LIME and Its Variants [Part1](LIME_Variants_part1.ipynb) (LIME) | [Part2](LIME_Variants_part2.ipynb) (GLIME). This tutorial introduces the LIME algorithms and many advanced improvements based on LIME.

- [GradCAM on Object Detection Models](GradCam_Object_Detection.ipynb). This tutorial shows how to use GradCAM to explain object detection models. [Mask-RCNN](https://arxiv.org/abs/1703.06870) and [PPYOLOE](https://arxiv.org/abs/2203.16250) are used as models.

- Transformers (to appear).

- Trustworthiness Evaluation Tutorials (to appear).

- Dataset-Level Tutorials (to appear).