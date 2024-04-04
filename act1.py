import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def tanh(x):
    return np.tanh(x)


x = np.linspace(-5, 5, 100)


y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)


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
