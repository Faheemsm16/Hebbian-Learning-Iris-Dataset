# Hebbian Neural Network for Iris Dataset Classification
This project demonstrates the implementation of a Hebbian Neural Network (HebbianNN) to classify the famous Iris dataset. The Iris dataset consists of 150 samples of iris flowers, classified into three species based on four features: sepal length, sepal width, petal length, and petal width. The objective of this project is to apply Hebbian learning, a form of unsupervised learning, to train the neural network and predict the species of the flowers.

## Dataset Overview: Iris Dataset
The Iris dataset is a well-known dataset in the machine learning community. It contains:
### Features:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width
### Target:
1. Setosa
2. Versicolor
3. Virginica
   
The dataset has 150 samples, with 50 samples for each of the three species. It is often used to demonstrate classification techniques.

## Model Overview: Hebbian Neural Network
A Hebbian Neural Network is an unsupervised learning model based on the Hebbian learning rule. The primary idea of this rule is that the weights between neurons are updated when both the pre- and post-synaptic neurons are active, following the formula: 
Δ𝑤 = 𝜂⋅𝑥⋅𝑦

Where:
- η is the learning rate
- x is the input feature
- y is the output neuron

In this implementation:
1. Activation function: Sigmoid function is used to introduce non-linearity.
2. Loss function: Mean Squared Error (MSE) is used to measure the discrepancy between the predicted and actual values.
3. Training process: The network updates its weights based on the Hebbian rule over a set number of epochs (iterations).

## Key Steps in the Code:
### 1. Data Preprocessing:
The Iris dataset is loaded and preprocessed by one-hot encoding the target values (species).
The features are standardized using StandardScaler to ensure that the model performs better by removing the scale differences between the features.
### 2. Hebbian Neural Network:
The network is initialized with random weights and uses the Hebbian learning rule to update the weights during training.
The network is trained for 10 epochs.
### 3. Loss Function:
During each epoch, the loss is computed using Mean Squared Error, and the weights are updated accordingly.
### 4. Model Output:
The output from the training process, showing the loss after each epoch, is as follows:

   Epoch 1, Loss: 0.2522
   
   Epoch 2, Loss: 0.2547
   
   Epoch 3, Loss: 0.2600
   
   Epoch 4, Loss: 0.2717
   
   Epoch 5, Loss: 0.2977
   
   Epoch 6, Loss: 0.3480
   
   Epoch 7, Loss: 0.4107
   
   Epoch 8, Loss: 0.4629
   
   Epoch 9, Loss: 0.5056
   
   Epoch 10, Loss: 0.5372

The loss decreases in the first few epochs but starts to increase toward the later epochs. This suggests that the network is having trouble minimizing the loss further, potentially due to limitations of the Hebbian learning approach, which is unsupervised and lacks the feedback mechanism of supervised learning like backpropagation.

## Conclusion:
This project provides an implementation of a Hebbian neural network to classify the Iris dataset. While the model demonstrates how Hebbian learning can be applied to a simple dataset, it also highlights the limitations of unsupervised learning in classification tasks when compared to more advanced supervised learning algorithms such as backpropagation-based neural networks.

This project offers insights into unsupervised learning and how neural networks can be trained without labeled feedback, providing a foundation for further exploration in neural network theory and applications.
