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
- **Activation Functions**
  - Understanding the commonly used activation functions and PyTorch implementation. Activation functions are used to introduce non-linearity in neural networks for better model learning.
- **Convolutional Neural Networks (CNNs)**
  - **Problem Solved**: Image classification
  - **Description**: Implemented a basic CNN using PyTorch to classify images from the CIFAR-10 dataset. The network consists of convolutional layers followed by fully connected layers, with ReLU activations and max pooling. The model is trained on the CIFAR-10 dataset and achieves accuracy results for each class.
- **Transfer Learning**
  - **Problem Solved**: Image classification with limited data
  - **Description**: Used a pretrained ResNet-18 model and fine-tuned it for a new classification task. The final fully connected layer of the pretrained model was replaced to match the number of classes in the new dataset. Two approaches were implemented:
    - **Finetuning the ConvNet**: The entire network was retrained on the new data, with all model parameters being updated.
    - **ConvNet as a Fixed Feature Extractor**: Only the final fully connected layer was trained, while all other layers were frozen to utilize the pretrained features without updating them.
- **TensorBoard**
  - **Problem Solved**: Visualizing training progress and model metrics
  - **Description**: Integrated TensorBoard to monitor the training process of a neural network on the MNIST dataset. TensorBoard was used to:
    - Visualize images from the dataset.
    - Track training loss and accuracy over time.
    - Add a computational graph of the model.
    - Evaluate the precision-recall curve for each class in the model's predictions.
- **Save and Load Models**
  - **Problem Solved**: Persisting models after training
  - **Description**: Demonstrated three different methods to save and load models in PyTorch. Key methods include:
    - `torch.save(arg, PATH)`: Saves the model, tensor, or dictionary to the specified path.
    - `torch.load(PATH)`: Loads the object stored at the specified path.
    - `torch.load_state_dict(arg)`: Loads only the state dictionary, which contains the model's parameters.
  - **Two Ways to Save Models**:
    - **Lazy Way (Save the Entire Model)**:
      - Save: `torch.save(model, PATH)`
      - Load: `model = torch.load(PATH)` and set the model to evaluation mode using `model.eval()`.
    - **Recommended Way (Save Only the State Dictionary)**:
      - Save: `torch.save(model.state_dict(), PATH)`
      - Load: Create the model instance and load the state dictionary using `model.load_state_dict(torch.load(PATH))`, then set it to evaluation mode using `model.eval()`.
  - **Loading Checkpoints**:
    - Save model and optimizer state dictionaries along with training metadata (e.g., epoch) for resuming training later.
    - Load the checkpoint and restore the model, optimizer, and any other necessary training state.
  - **Saving on Different Devices**:
    - Save on GPU and load on CPU.
    - Save on GPU and load on GPU.
    - Save on CPU and load on GPU, ensuring the model and input tensors are transferred to the appropriate device.
