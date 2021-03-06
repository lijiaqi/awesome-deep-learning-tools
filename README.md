# awesome-deep-learning-tools
collections of efficient tools for deep learning experiments, e.g., experiments control, hyperparameter optimizaiton

**Recommendations or PR are welcomed!**

## 0.Reading and Writing
- [https://paperswithcode.com/](https://paperswithcode.com/): paper with code

- [https://linggle.com/](https://linggle.com/): academic collocations

- [https://mathpix.com/](https://mathpix.com/): formular pictures to Latex code



## 1.Basic tools
- [logging](https://docs.python.org/3.6/library/logging.html): a python module for logging

- [tensorboard](https://www.tensorflow.org/tensorboard)(for TensorFlow)/[tensorboardX](https://github.com/lanpa/tensorboardX)(for PyTorch): visualization during experiments (**update**: PyTorch officially support TensorboardX since [v1.1.0](https://github.com/pytorch/pytorch/releases/tag/v1.1.0), please use `from torch.utils.tensorboard import SummaryWriter`.)

- [pyyaml](https://pyyaml.org/) or [ruamel.yaml](https://pypi.org/project/ruamel.yaml/): python modules for yaml configuration

## 2.For Experiment
### Project Template
- [Pytorch-Hydra-template](https://github.com/hobogalaxy/lightning-hydra-template): A clean and scalable template to kickstart your deep learning project
- [Pytorch-Project-Template](https://github.com/lijiaqi/PyTorch-Project-Template): **to be released**

### High Level API/Distributed Training
- [Pytorch-Lighting](https://github.com/PyTorchLightning/pytorch-lightning):The lightweight PyTorch wrapper for high-performance AI research.
- [apex](http://apex.run/): mixed-precisin (***no longer being maintainted***)
- [Horovod](https://eng.uber.com/horovod/) by Uber: a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.

### Configures Management
- [hydra](https://hydra.cc/)([github](https://github.com/facebookresearch/hydra)): A framework for elegantly configuring complex applications

- [yacs](https://github.com/rbgirshick/yacs): Yet Another Configuration System by [Ross Girshick](http://www.rossgirshick.info/)

### Data
- [alfred](https://github.com/jinfagang/alfred): A deep learning utility library for visualization and sensor fusion purpose

- [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html): A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications

### Experiments management
- Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
  - [Alchemy](https://github.com/catalyst-team/alchemy) - Experiments logging & visualization
  - [Catalyst](https://github.com/catalyst-team/catalyst) - Accelerated Deep Learning Research and Development
  - [Reaction](https://github.com/catalyst-team/reaction) - Convenient Deep Learning models serving
  
- [tensorboard.dev](https://tensorboard.dev/): visualization and tracking

- [wandb](https://www.wandb.com/): A tool for visualizing and tracking your machine learning experiments. 

- [fitlog](https://github.com/fastnlp/fitlog) by Fudan University: A tool for logging and code management

- [runx](https://github.com/NVIDIA/runx) by NVIDA: Deep Learning Experiment Management

- [NNI (Neural Network Intelligence)](https://nni.readthedocs.io/en/latest/Overview.html) by Microsoft: a toolkit to help users design and tune machine learning models (e.g., hyperparameters), neural network architectures, or complex system’s parameters, in an efficient and automatic way

- [TorchTracer](https://github.com/OIdiotLin/torchtracer/): a tool package for visualization and storage management in pytorch AI task.

### Hyperparameter Tuning
- [Tune](https://docs.ray.io/en/latest/tune.html): a Python library for experiment execution and hyperparameter tuning at any scale.
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization): A Python implementation of global optimization with gaussian processes.
- [adatune](https://github.com/awslabs/adatune): Gradient based Hyperparameter Tuning library in PyTorch
- [FAR-HO](https://github.com/lucfra/FAR-HO): Gradient based hyperparameter optimization & meta-learning package for TensorFlow
- [optuna](https://optuna.org/): An open source hyperparameter optimization framework to automate hyperparameter search

## 3. Libs
### Self-Supervised Learning
- [lightly](https://www.lightly.ai/)([github](https://github.com/lightly-ai/lightly)): a computer vision framework for self-supervised learning.
- [vissl](https://vissl.ai/)([github](https://github.com/facebookresearch/vissl)): Vision library for state-of-the-art Self-Supervised Learning research with PyTorch

### Semi-Supervised Learning && Domain Adaptation
- [salad](https://domainadaptation.org/)([github](https://github.com/domainadaptation/salad)): Semi-supervised Adaptive Learning Across Domains: a library for **domain adaptation**

### Domain Generalization
- [DomainBed](https://github.com/facebookresearch/DomainBed):a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434).
