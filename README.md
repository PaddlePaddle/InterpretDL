
[**中文**](./README_CN.md)

## InterpretDL: Interpretation of Deep Learning Models，基于『飞桨』的模型可解释性算法库

---

InterpretDL, short for *interpretation of deep learning models*, is a model interpretation toolkit for [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) models, and provides many interpretation algorithms, like LIME, Grad-CAM, IntergratedGradients, etc, for various tasks in CV and NLP.

*InterpretDL is under active construction !*

## Contents

* [Installation](#Installation)
* [Documentation](#Documentation)
* [Usage Guideline](#Usage-Guideline)
* [Contribution](#Contribution)
* [Roadmap](#Roadmap)
    * [Algorithms](#Algorithms)
    * [Tutorials](#Tutorials)
* [Copyright and License](#Copyright-and-License)

## Installation

Two steps:

1. Install [Paddle](https://www.paddlepaddle.org.cn/install/quick), recommendation with CUDA support.
2. `pip install interpretdl`.

## Documentation

Offline:
```
cd docs
make html
open _build/html/index.html
```

Online: To be appeared.

## Usage Guideline

All interpreters implement the abstract method `interpret()` from the abstract class `Interpreter`. So to use some interpreter, just create an instance of this interpreter, and then call `interpret()`.

More tutorials will be released.

## Roadmap

We are planning to create a useful toolkit for offering the model interpretation.

### Algorithms

We are planning to implement the algorithms below (categorized into sensitivity interpreters and algorithmic interpreters):

- [x] LIME
- [x] FastNormLIME
- [ ] NormLIME
- [x] LIMEPrior
- [ ] SmoothGrad
- [ ] DeepLIFT
- [ ] SHAP
- [x] GradCAM
- [x] IntegratedGradients
- [ ] InfluenceFunction
- [ ] ForgettingEvent
- [ ] SGDNoise
- [ ] More ...


### Tutorials
We plan to provide at least one example showing the usage for each interpretation algorithm.
While some algorithms can be widely applied, for those algorithms, we plan to give examples in different applications in CV, NLP etc.


## Copyright and License
InterpretDL is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/InterpretDL/blob/master/LICENSE).
