import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def tanh(x):
    return np.tanh(x)

# Generate input values
x = np.linspace(-5, 5, 100)

# Calculate outputs for each activation function
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU')
plt.title('ReLU Activation Function')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.title('Leaky ReLU Activation Function')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Tanh Activation Function')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

sigmoid_values = [sigmoid(val) for val in random_values]

print("Sigmoid values for random data:")
for val, sigmoid_val in zip(random_values, sigmoid_values):
    print(f"Input: {val}, Sigmoid: {sigmoid_val:.4f}")

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def tanh(x):
    return np.tanh(x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

relu_values = [relu(val) for val in random_values]
leaky_relu_values = [leaky_relu(val) for val in random_values]
tanh_values = [tanh(val) for val in random_values]

print("ReLU values for random data:")
for val, relu_val in zip(random_values, relu_values):
    print(f"Input: {val}, ReLU: {relu_val:.4f}")

print("\nLeaky ReLU values for random data:")
for val, leaky_relu_val in zip(random_values, leaky_relu_values):
    print(f"Input: {val}, Leaky ReLU: {leaky_relu_val:.4f}")

print("\nTanh values for random data:")
for val, tanh_val in zip(random_values, tanh_values):
    print(f"Input: {val}, Tanh: {tanh_val:.4f}")

