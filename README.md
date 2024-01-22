# Federated Learning with MNIST and HDC (Handwritten Digit Classification)

Welcome to the Federated Learning with MNIST and HDC project! This project explores the fascinating world of federated learning, a collaborative machine learning approach, using the MNIST dataset for handwritten digit classification. In addition, we incorporate the HDC (Handwritten Digit Classification) technique to enhance the model's performance.

## Overview

Federated learning enables multiple clients to collaboratively train a global model while keeping their data decentralized. In this project, we leverage the power of federated learning to train a handwritten digit classifier using the MNIST dataset. The models from different clients are then aggregated, and the HDC technique is applied to improve the overall accuracy.

## Project Steps

### Imports the Libraries

We start by importing essential libraries, including NumPy, TensorFlow's Keras, and scikit-learn. These libraries empower us to efficiently handle numerical operations, build neural networks, and manage data.

### Model / Data Parameters

Define the model and data parameters, such as the number of classes and input shape, setting the stage for the subsequent steps.

### Data Preparation

Load the MNIST dataset, scale images, and convert class vectors to binary matrices. These steps ensure the data is ready for training.

### Display an Image

Visualize a sample image from the training dataset using Matplotlib. Get a sneak peek into the handwritten digits you're about to classify.

### Splitting Training Data for Multiple Clients

Divide the training data into subsets for different clients, a key step in federated learning.

### Model Designing

Create a Convolutional Neural Network (CNN) model with HDC architecture. This architecture enhances the model's ability to capture intricate patterns in handwritten digits.

### Model Compilation and Training

Compile each client's model with suitable parameters and train them using their respective subsets of the training data.

### Model Evaluation on Test Data

Evaluate the performance of one of the models on the test set. Understand how well the model generalizes to unseen data.

### Model Aggregation and Weight Averaging

Aggregate predictions from each model, extract their weights, and calculate the average weights. This step ensures a collaborative learning experience.

### Handwritten Digit Classification

Visualize another image from the training dataset, reshape it, make predictions using the trained model, and display the predicted digit.

## Getting Started

1. Open the Jupyter Notebook `federated_learning_mnist_hdc.ipynb`.
2. Run each cell sequentially to observe the federated learning process and HDC in action.
3. Explore the model's performance on the test set and the effectiveness of model aggregation.

Feel free to experiment, tweak parameters, and expand on this project. The world of federated learning and handwritten digit classification is yours to explore!

Happy Coding! 🚀
