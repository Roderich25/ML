import numpy as np

# learning rate
alpha = 0.1
reg = 0.001
# XOR Logical Table
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Initial values
w_hly = np.random.uniform(size=(2, 2))  # 2 hidden layers with 1 neuron each
b_hly = [[-1, -1]]  # np.random.uniform(size=(1, 2))
w_out = np.random.uniform(size=(2, 1))
b_out = [-1]  # np.random.uniform(size=(1, 1))


def ActivationFunc(h):
    # return 1 / (1 + np.exp(-h))  # Sigmoid function
    return np.tanh(h)  # TanH


def ActivaitonFuncDerivative(h):
    # return h * (1 - h)  # Derivative of Sigmoid Function
    return 1 - h ** 2  # Derivative of TanH


def forwardProp(input, w_hly, b_hly, w_out, b_out):
    h_activation = np.dot(input, w_hly) + b_hly
    h_output = ActivationFunc(h_activation)
    y_activation = np.dot(h_output, w_out) + b_out
    y_output = ActivationFunc(y_activation)
    return h_output, y_output


def backwardProp(input, w_hly, b_hly, w_out, b_out, h_output, y_output):
    delta_out = (expected_output - y_output) * ActivaitonFuncDerivative(y_output)
    delta_hly = delta_out * w_out.T * ActivaitonFuncDerivative(h_output)
    w_out = w_out * np.array(1 - 2 * alpha * reg) + np.dot(h_output.T, delta_out) * alpha
    b_out = b_out * np.array(1 - 2 * alpha * reg) + np.sum(delta_out, axis=0, keepdims=True) * alpha
    w_hly = w_hly * np.array(1 - 2 * alpha * reg) + np.dot(input.T, delta_hly) * alpha
    b_hly = b_hly * np.array(1 - 2 * alpha * reg) + np.sum(delta_hly, axis=0, keepdims=True) * alpha
    return w_hly, b_hly, w_out, b_out


epochs = 0
while True:
    epochs += 1
    h_output, y_output = forwardProp(input, w_hly, b_hly, w_out, b_out)
    w_hly, b_hly, w_out, b_out = backwardProp(input, w_hly, b_hly, w_out, b_out, h_output, y_output)
    if np.sum(0.5 * (y_output - expected_output) ** 2) < 0.01 or epochs > 1_000_000:
        print(epochs)
        break

_, y_output = forwardProp(input, w_hly, b_hly, w_out, b_out)
print(y_output)
print(w_hly, b_hly, w_out, b_out)
