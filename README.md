# Deep Learning with PyTorch

This repository contains my implementations of deep learning concepts using PyTorch, following the tutorial by Patrick Loeber. The tutorial is an excellent resource for both beginners and those looking to solidify their understanding of deep learning with PyTorch. You can watch the full course [here](https://www.youtube.com/watch?v=c36lUUr864M&t=4425s).

## Environment

- **Python Version**: 3.12
- **Package Manager**: Anaconda

I am using Python 3.12, packaged by Anaconda, to manage the dependencies and environment for this project.

## Topics Covered

- **Linear Regression**: Implementing a simple linear regression model from scratch using PyTorch.
- **Logistic Regression**: Building a logistic regression model for binary classification tasks.
- **Dataset and DataLoader**: Understanding how to effectively manage and batch data for training deep learning models in PyTorch.
- **Transforms in PyTorch**: Applying transformations to data using PyTorch’s `torchvision.transforms` module.
- **Softmax and Cross-Entropy Loss**
  - **Softmax**
    - **Problem Solved**: Multi-class classification
    - **Function Type**: Activation Function
    - **Description**: Softmax converts logits (raw scores) into probabilities that sum to 1. It is used in multi-class classification problems where the model outputs a probability distribution over multiple classes.
  - **Cross-Entropy Loss**
    - **Problem Solved**: Multi-class classification
    - **Function Type**: Loss Function
    - **Description**: Cross-Entropy Loss measures the performance of a classification model whose output is a probability value between 0 and 1. The loss increases as the predicted probability diverges from the actual label.
- **Sigmoid**
  - **Problem Solved**: Binary classification
  - **Function Type**: Activation Function
  - **Description**: Sigmoid squashes the output to be between 0 and 1, making it suitable for binary classification tasks. It is used to convert raw output scores (logits) into probabilities for binary classification problems.