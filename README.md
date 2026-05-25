# MLP Implementation

## Introduction

A multilayer perceptron (MLP) is a type of neural network that processes data by passing it through several layers of connected nodes. Each layer learns patterns by adjusting weights and applying activation functions, which help the model understand complex relationships in the data (Zhang et al., 2023; Goodfellow et al., 2016).
In this project, I built an MLP using PyTorch to classify images from the Fashion-MNIST dataset, which contains different types of clothing. The model is based on concepts from Dive into Deep Learning and uses important techniques to improve performance, such as normalizing the input data, applying dropout to prevent overfitting, and using the Adam optimizer to efficiently update the model’s weights during training (Zhang et al., 2023; Goodfellow et al., 2016).

## Model Architecture

The model takes in 28×28 grayscale images and converts them into a single list of 784 values so they can be processed by the network. It then passes this data through two hidden layers. The first hidden layer has 256 neurons, and the second has 128 neurons, with both using ReLU activation functions to help the model learn more complex patterns (AIT Translation Hub, n.d.; Zhang et al., 2023; Goodfellow et al., 2016).
To reduce overfitting, dropout is applied after each hidden layer, randomly turning off some neurons during training with rates of 0.5 and 0.3. This helps the model generalize better to new data. The final output layer has 10 neurons, one for each clothing category in the Fashion-MNIST dataset, and produces scores (logits) used to determine the predicted class. Overall, this design represents a standard fully connected MLP and is more powerful than a model with only one hidden layer (Zhang et al., 2023; LeCun et al., 1998).

## Training Process and Optimization

The model is trained using mini-batch gradient descent with the Adam optimizer, which uses a learning rate of 0.001. Adam is an efficient optimization method that automatically adjusts how much each weight is updated, helping the model learn faster and more reliably (Goodfellow et al., 2016; Zhang et al., 2023). The model uses cross-entropy loss, which is commonly used for problems where the goal is to classify data into multiple categories.
During training, the model processes the data in small batches and calculates the average loss for each full pass through the dataset, known as an epoch. After each epoch, the model is tested on unseen data to measure its accuracy. The input data is normalized using the dataset’s mean and standard deviation to make training more stable and efficient. Dropout is applied during training to reduce overfitting, and it is automatically turned off during evaluation so the model can make consistent predictions (Zhang et al., 2023; Goodfellow et al., 2016).

## Key Concepts and Learning Outcomes

This implementation illustrates several key principles of neural network design. First, nonlinear activation functions like ReLU allow the network to learn complex relationships between inputs and outputs; without these functions, the network would behave like a simple linear model (Zhang et al., 2023; Goodfellow et al., 2016). Second, training with mini-batches improves computational efficiency and makes the gradient updates more stable. Third, dropout acts as a regularization method by randomly deactivating neurons during training, which helps prevent overfitting and encourages the network to learn more robust features (Zhang et al., 2023; Nielsen, 2015). Fourth, normalizing the input data ensures that features are on a consistent scale, which helps the optimizer converge faster and more reliably. Finally, using the Adam optimizer demonstrates how adaptive learning techniques can improve both training speed and model performance (Goodfellow et al., 2016; Zhang et al., 2023).

## Limitations and Potential Improvements

While the MLP performs reasonably well on the Fashion-MNIST dataset, it does not take advantage of the spatial relationships present in image data. Convolutional neural networks (CNNs) are typically more effective for image classification because they use local receptive fields and shared parameters, which often lead to higher accuracy (LeCun et al., 1998; Goodfellow et al., 2016). There are several ways this implementation could be improved. These include tuning hyperparameters such as the learning rate and batch size, experimenting with deeper or wider network architectures, adding batch normalization, tracking training accuracy in addition to test accuracy, and using data augmentation techniques to increase the diversity of the training dataset (Zhang et al., 2023; Nielsen, 2015).

## Conclusion

This project demonstrates a full implementation of a multilayer perceptron using PyTorch, covering data preprocessing, model construction, training, and evaluation. By combining two hidden layers, ReLU activation functions, dropout regularization, and the Adam optimizer, this implementation highlights how fundamental components of neural networks work together to achieve effective learning and generalization (Zhang et al., 2023; Goodfellow et al., 2016). The work also provides a solid foundation for future exploration of more advanced architectures, such as convolutional neural networks, which can better capture spatial features in image data (LeCun et al., 1998; Zhang et al., 2023).

# References

AIT Translation Hub. (n.d.). Deep learning neuron. https://aitranslationhub.com/deep-neural-network/deep-learning-neuron/
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.
Nielsen, M. A. (2015). Neural networks and deep learning. Determination Press.
Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press.
