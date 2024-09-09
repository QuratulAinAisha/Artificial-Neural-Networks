# Perceptron Model for OR Gate Classification

This repository contains a simple implementation of a **Perceptron** in Python. The perceptron is trained to simulate the behavior of an OR gate using binary inputs. The Perceptron model is a basic neural network structure that performs binary classification.

## Features

- **Perceptron Implementation**: Includes a `Perceptron` class that handles weight initialization, training (using the perceptron learning rule), and predictions.
- **Activation Function**: Uses a step function as the activation function.
- **Training with OR Gate**: Trains the perceptron to act as an OR gate, with input combinations of `[0, 0]`, `[0, 1]`, `[1, 0]`, and `[1, 1]`.
- **Training Process Visualization**: During the training process, the state of the perceptron (weights, bias, and predictions) is printed for each epoch.

## Getting Started

### Prerequisites

This project requires the following Python libraries:

- `numpy`

You can install the required packages by running:

```bash
pip install numpy
```

### Installation

1. Clone this repository:

```bash
git clone https://github.com/QuratulainAisha/Artificial-Neural-Network.git
cd Artificial-Neural-Network
```

2. Install the dependencies:

```bash
pip install numpy
```

### Running the Code

You can run the perceptron model by executing the `main.py` file. The perceptron will be trained on an OR gate dataset and make predictions after training.

```bash
python main.py
```

### Example Output

During training, the perceptron will update its weights and bias. You'll see the training progress printed to the console for each epoch. Once the training is complete, the perceptron will make predictions for all input combinations.

Example output:

```
Epoch 1/10:
  Inputs: [0 0], Prediction: 0, True Label: 0, Error: 0
  Updated Weights: [0. 0.], Updated Bias: 0.0
  Inputs: [0 1], Prediction: 0, True Label: 1, Error: 1
  Updated Weights: [0.  0.1], Updated Bias: 0.1
  ...
Final Predictions:
  Inputs: [0 0], Prediction: 0
  Inputs: [0 1], Prediction: 1
  Inputs: [1 0], Prediction: 1
  Inputs: [1 1], Prediction: 1
```

## Code Explanation

### `Perceptron` Class

- **`__init__(self, input_size, learning_rate=0.1, epochs=10)`**: Initializes the perceptron with the given number of input features, learning rate, and the number of epochs for training.
  
- **`activation(self, x)`**: Implements the step function (activation function) which outputs `1` if the input is >= 0 and `0` otherwise.

- **`predict(self, x)`**: Computes the linear combination of input weights and bias, and applies the activation function to predict the output.

- **`fit(self, X, y)`**: Trains the perceptron on the provided data (`X` is the input, `y` is the target output) using the perceptron learning rule. It also prints the weights, bias, and error for each input during training.

### OR Gate Dataset

The perceptron is trained to mimic the behavior of an OR gate with the following dataset:

| Input (X1, X2) | Output (Y) |
|----------------|------------|
| (0, 0)         | 0          |
| (0, 1)         | 1          |
| (1, 0)         | 1          |
| (1, 1)         | 1          |

The model should learn to predict the correct OR gate output after training.

## How It Works

- **Step Function**: The perceptron uses a simple step function as an activation function. It outputs `1` if the weighted sum of inputs plus the bias is greater than or equal to zero, and `0` otherwise.

- **Learning Rule**: The perceptron updates its weights and bias after each prediction based on the error (the difference between the predicted output and the actual output).

- **Training**: During training, the perceptron adjusts its weights and bias over multiple epochs until it correctly classifies all training examples.


## Acknowledgments

This project is inspired by the classic implementation of the perceptron for binary classification tasks like the OR gate, which is one of the foundational concepts in machine learning and neural networks.
