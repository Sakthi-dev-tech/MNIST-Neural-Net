# MNIST Neural Net

A simple handwritten digit classifier for the MNIST dataset built from scratch with
NumPy. The project does not use PyTorch or TensorFlow; the forward pass,
backpropagation, and gradient descent update rules are implemented directly in
`model/main.ipynb`.

## Project Structure

```text
MNIST-Neural-Net/
├── data/
│   └── train.csv
├── model/
│   └── main.ipynb
├── requirements.txt
└── README.md
```

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then open and run `model/main.ipynb` in VS Code, JupyterLab, or Jupyter
Notebook. If you want to launch it from the terminal and do not already have
Jupyter installed, install it first:

```bash
pip install notebook
```

Then start Jupyter Notebook:

```bash
jupyter notebook model/main.ipynb
```

The required packages are:

- `numpy`
- `pandas`
- `matplotlib`

## Dataset

The notebook uses `data/train.csv`, which follows the Kaggle MNIST CSV format.
Each row contains:

- `label`: the correct digit from `0` to `9`
- `pixel0` to `pixel783`: grayscale pixel values for a `28 x 28` image

The raw pixel values are in the range `0` to `255`. The notebook normalizes them
by dividing by `255`, so every input feature is scaled to the range `0` to `1`.

The data is shuffled and split as follows:

- Validation set: first `1,000` examples
- Training set: remaining `41,000` examples

The feature matrix is transposed so each column is one image:

```text
X_train shape = (784, 41000)
X_val shape   = (784, 1000)
```

## Model Architecture

The neural network has one hidden layer:

```text
Input layer:   784 values, one for each image pixel
Hidden layer:  10 neurons with ReLU activation
Output layer:  10 neurons with softmax activation
```

The output layer gives a probability-like score for each digit class from `0` to
`9`. The predicted digit is the index with the highest output value.

Parameter shapes:

```text
W1: (10, 784)
b1: (10, 1)
W2: (10, 10)
b2: (10, 1)
```

The parameters are initialized randomly in the range `[-0.5, 0.5)`.

## Forward Propagation

For an input matrix `X`, the notebook computes the first layer linear output:

```text
Z1 = W1X + b1
```

The hidden layer activation uses ReLU:

```text
A1 = ReLU(Z1)
ReLU(Z) = max(0, Z)
```

The second layer linear output is:

```text
Z2 = W2A1 + b2
```

The output activation uses softmax:

```text
A2 = softmax(Z2)
softmax(Z_i) = exp(Z_i) / sum(exp(Z_j))
```

In the implementation, `Z2` is shifted by subtracting the column-wise maximum
before applying `exp`. This improves numerical stability without changing the
softmax result.

## One-Hot Encoding

The labels are converted into one-hot vectors. For example, if the correct label
is `4`, the target vector has a `1` at index `4` and `0` everywhere else:

```text
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
```

This creates a target matrix `Y_one_hot` with shape:

```text
(10, number_of_examples)
```

## Backpropagation

The output layer error is:

```text
dZ2 = A2 - Y_one_hot
```

The gradients for the second layer are:

```text
dW2 = (1 / m) dZ2 A1.T
db2 = (1 / m) sum(dZ2)
```

The hidden layer error is:

```text
dZ1 = W2.T dZ2 * ReLU'(Z1)
```

where:

```text
ReLU'(Z) = 1 if Z > 0, else 0
```

The gradients for the first layer are:

```text
dW1 = (1 / m) dZ1 X.T
db1 = (1 / m) sum(dZ1)
```

Here, `m` is the number of training examples.

## Gradient Descent

After each forward and backward pass, the parameters are updated with gradient
descent:

```text
W1 = W1 - alpha dW1
b1 = b1 - alpha db1
W2 = W2 - alpha dW2
b2 = b2 - alpha db2
```

In the saved notebook run:

```text
iterations = 500
alpha = 0.1
```

The notebook prints training accuracy every `50` iterations. In the saved output,
the model reaches about `84.5%` training accuracy by iteration `450`.

## Prediction Process

To make a prediction, the notebook runs forward propagation with the trained
parameters:

```text
_, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
prediction = argmax(A2)
```

The `test_predictions` helper selects an image from the training set, predicts
its label, prints the prediction and true label, then reshapes the `784` input
features back into a `28 x 28` image for display with Matplotlib.

## Current Notes

This project is intended as a learning implementation of a neural network. It
shows the full training pipeline manually:

1. Load CSV data with pandas.
2. Convert data to NumPy arrays.
3. Shuffle and split the dataset.
4. Normalize image pixels.
5. Initialize weights and biases.
6. Run forward propagation.
7. Compute gradients with backpropagation.
8. Update parameters with gradient descent.
9. Measure accuracy.
10. Visualize sample predictions.
