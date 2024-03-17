# PyTorch Tutorial
![pytorch_logo_icon_169823](https://github.com/galax19ksh/PyTorch-Tutorial/assets/112553872/2c70362f-9894-4351-ab34-31360d85b1d1)

Welcome to the PyTorch Tutorial repository! This repository contains a collection of tutorials and code examples to help you learn and understand PyTorch, a popular deep learning framework.

## Introduction
[PyTorch](https://pytorch.org/) is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and dynamic computational graph, allowing for easy experimentation and efficient deployment of deep learning models. This tutorial repository is designed to guide beginners through the basics of PyTorch and help them build a strong foundation in deep learning concepts and practices.

### Official Documentation
Refer to the official [PyTorch Documentation](https://pytorch.org/docs/stable/index.html).

### Built in Datasets in PyTorch

* **[Image Datasets](https://pytorch.org/vision/stable/datasets.html)**: They are used  for image classification, Image detection, optical flow, stereo matching, image captioning, video classification and prediction etc. They are available in `torchvision.datasets` module.
* **[Text Datasets](https://pytorch.org/text/stable/datasets.html)**: They are for text classification, language modeling, machine translation, sequence tagging, question answer and supervised learning. They can be accessed by the module `torchtext.datasets`.
* **[Audio Datasets](https://pytorch.org/audio/stable/datasets.html)**: They can be accessed using the module `torchaudio.datasets`.
 

## Pytorch VS Tensorflow
They are the two most popular deep learning libraries within the machine learning community. Both PyTorch and TensorFlow are powerful frameworks with their own strengths and weaknesses. The choice between them often depends on factors such as ease of use, deployment requirements, community support, and personal preference.

| Feature                    | PyTorch                                     | TensorFlow                                  |
|----------------------------|---------------------------------------------|---------------------------------------------|
|Developed By | Meta (Facebook) AI | Google Brain |
|Computational Graph        | Dynamic                                     | Static (with eager execution in TensorFlow 2.x)|
| Ease of Use                | Pythonic API, intuitive                     | Historically complex, improved with TensorFlow 2.x|
| Model Deployment           | TorchServe, TorchScript                     | TensorFlow Serving, TensorFlow Lite, TensorFlow.js|
| Community and Ecosystem    | Growing rapidly, strong research community | Mature ecosystem, strong industry support    |
| Debugging and Visualization| Standard Python debugging tools, PyTorch Lightning | TensorBoard for visualization, TensorFlow Debugging API |
| Integration with Libraries| NumPy, scikit-learn                         | TensorFlow Probability, TensorFlow Data Validation |
| Scalability | Less | Better |
| Applications | Research-oriented | Industry-oriented |


## Installation
To get started with PyTorch, you'll first need to install it on your system. You can install PyTorch using pip or conda, depending on your preference and system configuration. Follow the instructions on the official [PyTorch website](https://pytorch.org/) to install PyTorch on your system. Personality I prefer [Google Colab](https://colab.research.google.com/) most of the time due to lack of computational power of my system.


## Tutorials 
Check out the detailed tuturials [here](https://pytorch.org/tutorials/).
This repo is mainly for the small projects for practice to get started with PyTorch for deep learning.

* **Tensors:** Learn the fundamentals of PyTorch with tensors, the building blocks for data representation and computation in deep learning.

* **Datasets & DataLoaders:** Explore how to efficiently load and process data using PyTorch's Datasets and DataLoaders, simplifying the data handling pipeline for training and evaluation. PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`.

* **Transforms:** Understand the power of data augmentation and preprocessing with PyTorch's Transforms, facilitating flexible and customizable data manipulation for improved model performance. Using the module `torchvision.transforms` offers several commonly used transforms out of the box.

* **Build Model:** Dive into building neural network models effortlessly with PyTorch, leveraging its intuitive and flexible API for constructing complex architectures. After importing the library using `from torch import nn`, we build the blocks and then add layers like `nn.Flatten`, `nn.ReLU`, `nn.Conv2d`, `nn.Linear`, `nn.Sequential`, `nn.Softmax` etc.

* **Autograd:** Harness the automatic differentiation capabilities of PyTorch's Autograd, enabling dynamic computation of gradients for efficient training of deep learning models. Back propagation is the most frequently used algorithm. `torch.autograd` is a built-in differentiation engine that supports automatic computation of gradient for any computational graph.

* **Optimization:** Master the art of optimizing neural network parameters with PyTorch's Optimization tools, including a variety of optimization algorithms for fine-tuning model performance. Check out `nn.<loss>` and `torch.optim` functions.

* **Save & Load Model:** Learn how to save and load trained models in PyTorch, ensuring seamless deployment and sharing of deep learning models for real-world applications. `torch.save(model, 'model.pth')` for saving model and `model = torch.load('model.pth')` for loading the model.

