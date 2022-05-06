[**中文**](./README_CN.md) | English

[![Release](https://img.shields.io/github/release/PaddlePaddle/InterpretDL.svg)](https://github.com/PaddlePaddle/InterpretDL/releases)
[![PyPI](https://img.shields.io/pypi/v/interpretdl.svg)](https://pypi.org/project/interpretdl)
[![CircleCI](https://circleci.com/gh/PaddlePaddle/InterpretDL.svg?style=shield)](https://circleci.com/gh/PaddlePaddle/InterpretDL)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://interpretdl.readthedocs.io/en/latest/index.html)
[![Downloads](https://static.pepy.tech/personalized-badge/interpretdl?period=total&units=abbreviation&left_color=grey&right_color=green&left_text=Downloads%20Total)](https://pepy.tech/project/interpretdl)


# InterpretDL: Interpretation of Deep Learning Models based on PaddlePaddle

InterpretDL, short for *interpretations of deep learning models*, is a model interpretation toolkit for [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) models. This toolkit contains implementations of many interpretation algorithms, including LIME, Grad-CAM, Integrated Gradients and more. Some SOTA and new interpretation algorithms are also implemented.

*InterpretDL is under active construction and all contributions are welcome!*

![](https://user-images.githubusercontent.com/13829174/159609890-bf3f2050-1ee8-482f-baac-693d280f9039.jpg)

# Why InterpretDL

The increasingly complicated deep learning models make it impossible for people to understand their internal workings. Interpretability of black-box models has become the research focus of many talented researchers. InterpretDL provides a collection of both classical and new algorithms for interpreting models.

By utilizing these helpful methods, people can better understand why models work and why they don't, thus contributing to the model development process.

For researchers working on designing new interpretation algorithms, InterpretDL gives an easy access to existing methods that they can compare their work with.

# :fire: :fire: :fire: News :fire: :fire: :fire:

- (2022/04/27) A getting-started tutorial is provided. Check it from [GitHub](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/Getting_Started.ipynb) or [NBViewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/Getting_Started.ipynb). Usage examples have been provided for each algorithm (both Interpreter and Evaluator). We are currently preparing tutorials for easy usages of InterpretDL. Both tutorials and examples can be assessed under the [tutorial](https://github.com/PaddlePaddle/InterpretDL/tree/master/tutorials) folder.

- (2022/01/06) Implemented the Cross-Model Consensus Explanation method. In brief, this method averages the explanation results from several models. Instead of interpreting individual models, this method is able to identify the discriminative features in the input data with accurate localization. See the [paper](https://arxiv.org/abs/2109.00707) for details.

  * `Consensus`: Xuhong Li, Haoyi Xiong, Siyu Huang, Shilei Ji, Dejing Dou. Cross-Model Consensus of Explanations and Beyond for Image Classification Models: An Empirical Study. arXiv:2109.00707.

We show a demo with six models (the last column shows the consensus explanation), while more models (around 15) could give a much better result. See the [example](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_consensus_cv.ipynb) for more details.

![Consensus Result](https://user-images.githubusercontent.com/13829174/165700043-1c680494-8573-4b4a-a2d6-74ea3d14f214.png)

# Demo

Interpretation algorithms give a hint of why a black-box model makes its decision.

The following table gives visualizations of several interpretation algorithms applied to the original image to tell us why the model predicts "bull_mastiff."
| Original Image | IntGrad ([demo](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_int_grad_cv.ipynb)) | SG ([demo](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_smooth_grad_cv.ipynb)) | LIME ([demo](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_lime_cv.ipynb)) | Grad-CAM ([demo](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_grad_cam_cv.ipynb)) |
:-----------:|:-----------:|:-----------:|:-----------:|:-----------:
![](imgs/catdog.jpg)|![](imgs/catdog_ig_overlay.jpeg)|![](imgs/catdog_sg_overlay.jpeg)|![](imgs/catdog_lime_overlay.jpeg)|![](imgs/catdog_gradcam_overlay.jpeg)

For sentiment analysis task, the reason why a model gives positive/negative predictions can be visualized as follows. A quick demo can be found [here](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-2.0-en-sst-2.ipynb). Samples in Chinese are also available [here](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/ernie-1.0-zh-chnsenticorp.ipynb).

![](imgs/sentiment-en.png)


# Contents

- [InterpretDL: Interpretation of Deep Learning Models based on PaddlePaddle](#interpretdl-interpretation-of-deep-learning-models-based-on-paddlepaddle)
- [Why InterpretDL](#why-interpretdl)
- [:fire: :fire: :fire: News :fire: :fire: :fire:](#fire-fire-fire-news-fire-fire-fire)
- [Demo](#demo)
- [Contents](#contents)
- [Installation](#installation)
  - [Pip installation](#pip-installation)
  - [Developer installation](#developer-installation)
  - [Unit Tests](#unit-tests)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Examples and Tutorials](#examples-and-tutorials)
- [Roadmap](#roadmap)
  - [Implemented Algorithms with Taxonomy](#implemented-algorithms-with-taxonomy)
  - [Implemented Trustworthiness Evaluation Algorithms](#implemented-trustworthiness-evaluation-algorithms)
- [Presentations](#presentations)
- [References of Algorithms](#references-of-algorithms)
- [Copyright and License](#copyright-and-license)
- [Recent News](#recent-news)

# Installation

It requires the deep learning framework [paddlepaddle](https://www.paddlepaddle.org.cn/install/quick), versions with CUDA support are recommended.

## Pip installation

```bash
pip install interpretdl

# or with tsinghua mirror
pip install interpretdl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Developer installation

```bash
git clone https://github.com/PaddlePaddle/InterpretDL.git
# ... fix bugs or add new features
cd InterpretDL && pip install -e .
# welcome to propose pull request and contribute
yapf -i <python_file_path>  # code style: column_limit=120
```

## Unit Tests

```bash
# run gradcam unit tests
python -m unittest -v tests.interpreter.test_gradcam
# run all unit tests
python -m unittest -v
```

# Documentation

Online link: [interpretdl.readthedocs.io](https://interpretdl.readthedocs.io/en/latest/index.html).

Or generate the docs locally:

```bash
git clone https://github.com/PaddlePaddle/InterpretDL.git
cd docs
make html
open _build/html/index.html
```

# Getting Started

All interpreters inherit the abstract class [`Interpreter`](https://github.com/PaddlePaddle/InterpretDL/blob/4f7444160981e99478c26e2a52f8e40bd06bf644/interpretdl/interpreter/abc_interpreter.py), of which `interpret(**kwargs)` is the function to call.

```python
# an example of SmoothGradient Interpreter.

import interpretdl as it
from paddle.vision.models import resnet50
paddle_model = resnet50(pretrained=True)
sg = it.SmoothGradInterpreter(paddle_model, use_cuda=True)
gradients = sg.interpret("test.jpg", visual=True, save_path=None)
```

A quick [Getting-Started tutorial](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/Getting_Started.ipynb) (or [on NBviewer](https://nbviewer.org/github/PaddlePaddle/InterpretDL/blob/master/tutorials/Getting_Started.ipynb)) is provided. It takes only a few minutes to be familiar with InterpretDL.

# Examples and Tutorials

We have provided at least one example for each interpretation algorithm and each trustworthiness evaluation algorithm, hopefully covering applications for both CV and NLP.

We are currently preparing tutorials for easy usages of InterpretDL.

Both examples and tutorials can be accessed under [tutorials](https://github.com/PaddlePaddle/InterpretDL/tree/master/tutorials) folder.

# Roadmap

We are planning to create a useful toolkit for offering the model interpretations as well as evaluations.
We have now implemented the interpretation algorithms as follows, and we are planning to add more algorithms that are desired.
Welcome to contribute or just tell us which algorithms are desired.

## Implemented Algorithms with Taxonomy

Two dimensions (representations of explanation results and types of the target model) are used to categorize the interpretation algorithms. This taxonomy can be an indicator to find the best suitable algorithm for the target task and model.

| Methods                         | Representation        | Model Type                                     |
|---------------------------------|-----------------------|------------------------------------------------|
| [LIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lime.py)                            | Input Features        | Model-Agnostic                                 |
| [LIME with Prior](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lime_prior.py)                 | Input Features        | Model-Agnostic                                 |
| [NormLIME/FastNormLIME](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/_normlime_base.py)           | Input Features        | Model-Agnostic                                 |
| [LRP](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/lrp.py)                             | Input Features        | Differentiable* |
| [SmoothGrad](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/smooth_grad.py)                      | Input Features        | Differentiable                                 |
| [IntGrad](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/integrated_gradients.py)                         | Input Features        | Differentiable                                 |
| [GradSHAP](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/gradient_shap.py)                        | Input Features        | Differentiable                                 |
| [Occlusion](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/occlusion.py)                     | Input Features        | Model-Agnostic                                 |
| [GradCAM/CAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/gradient_cam.py)                     | Intermediate Features | Specific: CNNs                                 |
| [ScoreCAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/score_cam.py)                        | Intermediate Features | Specific: CNNs                                 |
| [Rollout](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/rollout.py)                         | Intermediate Features | Specific: Transformers                         |
| [TAM](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/transition_attention_maps.py)                             | Intermediate Features | Specific: Transformers                         |
| [ForgettingEvents](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/forgetting_events.py)                | Dataset-Level         | Differentiable                                 |
| [TIDY (Training Data Analyzer)](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/TIDY.ipynb) | Dataset-Level         | Differentiable                                 |
| [Consensus](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/consensus.py)                       | Features              | Cross-Model                                    |
| [Generic Attention](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/interpreter/generic_attention.py)                  | Input Features                | Specific: Bi-Modal Transformers |

\* LRP requires that the model is of specific implementations for relevance back-propagation.

## Implemented Trustworthiness Evaluation Algorithms

- [x] [Perturbation Tests](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/perturbation.py)
- [x] [Deletion & Insertion](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/deletion_insertion.py)
- [x] [Localization Ability](https://github.com/PaddlePaddle/InterpretDL/blob/master/interpretdl/evaluate_interpreter/localization.py)

## Planning Alorithms

* Intermediate Features Interpretation Algorithm
  - [ ] More Transformers Specific Interpreters

* Dataset-Level Interpretation Algorithms
  - [ ] Influence Function

* Evaluations
  - [ ] Local Fidelity
  - [ ] Sensitivity

# Presentations
**Linux Foundation Project AI & Data** -- Interpretable Deep Learning: Interpretation, Interpretability, Trustworthiness, and Beyond. [Video Link](https://wiki.lfaidata.foundation/download/attachments/7733341/GMT20220324-130226_Recording_3840x2160.mp4?version=1&modificationDate=1649079184753&api=v2) (00:20:30 -- 00:45:00).

**Baidu Create 2021 (in Chinese)**: [Video Link](https://live.baidu.com/m/media/pclive/pchome/live.html?room_id=5073321791&source=h5pre) (01:18:40 -- 01:36:30).

**ICML 2021 Expo** -- Interpretable Deep Learning: Interpretation, Interpretability, Trustworthiness, and Beyond. [Video Link](https://icml.cc/ExpoConferences/2021/workshop/11429#wse-detail-11435).

# References of Algorithms

* `SGDNoise`: [On the Noisy Gradient Descent that Generalizes as SGD, Wu et al 2019](https://arxiv.org/abs/1906.07405)
* `IntegratedGraients`: [Axiomatic Attribution for Deep Networks, Mukund Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)
* `CAM`, `GradCAM`: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Ramprasaath R. Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391.pdf)
* `SmoothGrad`: [SmoothGrad: removing noise by adding noise, Daniel Smilkov et al. 2017](https://arxiv.org/abs/1706.03825)
* `GradientShap`: [A Unified Approach to Interpreting Model Predictions, Scott M. Lundberg et al. 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
* `Occlusion`: [Visualizing and Understanding Convolutional Networks, Matthew D Zeiler and Rob Fergus 2013](https://arxiv.org/abs/1311.2901)
* `Lime`: ["Why Should I Trust You?": Explaining the Predictions of Any Classifier, Marco Tulio Ribeiro et al. 2016](https://arxiv.org/abs/1602.04938)
* `NormLime`: [NormLime: A New Feature Importance Metric for Explaining Deep Neural Networks, Isaac Ahern et al. 2019](https://arxiv.org/abs/1909.04200)
* `ScoreCAM`: [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks, Haofan Wang et al. 2020](https://arxiv.org/abs/1910.01279)
* `ForgettingEvents`: [An Empirical Study of Example Forgetting during Deep Neural Network Learning, Mariya Toneva et al. 2019](http://arxiv.org/abs/1812.05159)
* `LRP`: [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation, Bach et al. 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
* `Rollout`: [Quantifying Attention Flow in Transformers, Abnar et al. 2020](https://arxiv.org/abs/2005.00928)
* `TAM`: [Explaining Information Flow Inside Vision Transformers Using Markov Chain. Yuan et al. 2021](https://openreview.net/forum?id=TT-cf6QSDaQ)
* `Consensus`: [Cross-Model Consensus of Explanations and Beyond for Image Classification Models: An Empirical Study. Li et al 2021](https://arxiv.org/abs/2109.00707)
* `Perturbation`: [Evaluating the visualization of what a deep neural network has learned.](https://arxiv.org/abs/1509.06321)
* `Deletion&Insertion`: [RISE: Randomized Input Sampling for Explanation of Black-box Models.](https://arxiv.org/abs/1806.07421)
* `PointGame`: [Top-down Neural Attention by Excitation Backprop.](https://arxiv.org/abs/1608.00507)
* `Generic Attention`: [Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers
](https://arxiv.org/abs/2103.15679)

# Copyright and License

InterpretDL is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/InterpretDL/blob/master/LICENSE).

# Recent News


- (2021/10/20) Implemented the Transition Attention Maps (TAM) explanation method for PaddlePaddle [Vision Transformers](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/model_zoo/vision_transformer.py). As always, several lines call this interpreter. See details from the [example](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_tam_cv_ViT.ipynb), and the [paper](https://openreview.net/forum?id=TT-cf6QSDaQ):

  * `TAM`: Tingyi Yuan, Xuhong Li, Haoyi Xiong, Hui Cao, Dejing Dou. Explaining Information Flow Inside Vision Transformers Using Markov Chain. In *Neurips 2021 XAI4Debugging Workshop*. 

```python
import paddle
import interpretdl as it

# load vit model and weights
# !wget -c https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams -P assets/
from assets.vision_transformer import ViT_base_patch16_224
paddle_model = ViT_base_patch16_224()
MODEL_PATH = 'assets/ViT_base_patch16_224_pretrained.pdparams'
paddle_model.set_dict(paddle.load(MODEL_PATH))

# Call the interpreter.
tam = it.TAMInterpreter(paddle_model, use_cuda=True)
img_path = 'samples/el1.png'
heatmap = tam.interpret(
        img_path,
        start_layer=4,
        label=None,  # elephant
        visual=True,
        save_path=None)
heatmap = tam.interpret(
        img_path,
        start_layer=4,
        label=340,  # zebra
        visual=True,
        save_path=None)
```
| image | elephant | zebra |
:-----------:|:-----------:|:-----------:
![image](https://user-images.githubusercontent.com/13829174/139223230-66094dbf-cbc8-450c-acd8-0c0ec40c5fef.png) | ![elephant](https://user-images.githubusercontent.com/13829174/138049903-8106d879-3c70-437b-a580-cf8e9c17f974.png) | ![zebra](https://user-images.githubusercontent.com/13829174/138049895-6d52b97d-c4fd-40da-be88-f5c956cb9fcb.png)



- (2021/07/22) Implemented Rollout Explanations for PaddlePaddle [Vision Transformers](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/model_zoo/vision_transformer.py). See the [notebook](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/example_rollout_cv_ViT.ipynb) for the visualization.

```python
import paddle
import interpretdl as it

# wget -c https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams -P assets/
from assets.vision_transformer import ViT_small_patch16_224
paddle_model = ViT_small_patch16_224()
MODEL_PATH = 'assets/ViT_small_patch16_224_pretrained.pdparams'
paddle_model.set_dict(paddle.load(MODEL_PATH))

img_path = 'assets/catdog.png'
rollout = it.RolloutInterpreter(paddle_model, use_cuda=True)
heatmap = rollout.interpret(img_path, start_layer=0, visual=True)
```
