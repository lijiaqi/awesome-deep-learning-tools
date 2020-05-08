# awesome-deep-learning-tools
collections of efficient tools for deep learning experiments, e.g., experiments control, hyperparameter optimizaiton

**Recommendations or PR are welcomed!**

## 1.Basic tools
- [logging](https://docs.python.org/3.6/library/logging.html): a python module for logging

- [tensorboard](https://www.tensorflow.org/tensorboard)(for TensorFlow)/[tensorboardX](https://github.com/lanpa/tensorboardX)(for PyTorch): visualization during experiments

- [pyyaml](https://pyyaml.org/) or [ruamel.yaml](https://pypi.org/project/ruamel.yaml/): python modules for yaml configuration

- [yacs](https://github.com/rbgirshick/yacs): Yet Another Configuration System by [Ross Girshick](http://www.rossgirshick.info/)

## 2.Higher phrase
### Data
- [alfred](https://github.com/jinfagang/alfred): A deep learning utility library for visualization and sensor fusion purpose

- [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html): A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications

### Experiments management
- [wandb](https://www.wandb.com/): A tool for visualizing and tracking your machine learning experiments. 

- [fitlog](https://github.com/fastnlp/fitlog) by Fudan University: A tool for logging and code management

- [runx](https://github.com/NVIDIA/runx) by NVIDA: Deep Learning Experiment Management

- [NNI (Neural Network Intelligence)](https://nni.readthedocs.io/en/latest/Overview.html) by Microsoft: a toolkit to help users design and tune machine learning models (e.g., hyperparameters), neural network architectures, or complex system’s parameters, in an efficient and automatic way

### Hyperparameter Optimization
- [Tune](https://docs.ray.io/en/latest/tune.html): a Python library for experiment execution and hyperparameter tuning at any scale.
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization): A Python implementation of global optimization with gaussian processes.
- [adatune](https://github.com/awslabs/adatune): Gradient based Hyperparameter Tuning library in PyTorch
- [FAR-HO](https://github.com/lucfra/FAR-HO): Gradient based hyperparameter optimization & meta-learning package for TensorFlow

### High Level API/Distributed Training
- [pytorch-lighting](https://pytorch-lightning.readthedocs.io/en/stable/): a higher level API for PyTorch
- [apex](http://apex.run/): mixed-precisin (***no longer being maintainted***)
- [Horovod](https://eng.uber.com/horovod/) by Uber: a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
