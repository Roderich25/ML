import numpy as np

# learning rate
alpha = 0.1

# XOR Logical Table
x = np.array([-1, 1, 1])
y = np.array([0])

# Initial values
w_h1 = np.array([0.8, 0.5, 0.4])
w_h2 = np.array([-0.1, 0.9, 1.0])
w_out = np.array([0.3, -1.2, 1.1])


def ActivationFunc(h):
    # return 1 / (1 + np.exp(-h))  # Sigmoid function
    return np.tanh(h)  # TanH


def ActivaitonFuncDerivative(h):
    # return h * (1 - h)  # Derivative of Sigmoid Function
    return 1 - h ** 2  # Derivative of TanH


def ForwardProp(x, w_h1, w_h2, w_out):
    h1 = ActivationFunc(np.dot(x, w_h1))
    h2 = ActivationFunc(np.dot(x, w_h2))
    output = ActivationFunc(np.dot(np.append(-1, [h1, h2]), w_out))
    return h1, h2, output


def BackwardProp(y, x, w_h1, w_h2, w_out):
    h1, h2, output = ForwardProp(x, w_h1, w_h2, w_out)
    error = y - output

    delta_out = error * ActivaitonFuncDerivative(output)
    w_out += alpha * np.append(-1, [h1, h2]) * delta_out

    delta_h1 = ActivaitonFuncDerivative(h1) * delta_out * w_out[1]
    w_h1 += alpha * x * delta_h1

    delta_h2 = ActivaitonFuncDerivative(h2) * delta_out * w_out[2]
    w_h2 += alpha * x * delta_h2

    return w_h1, w_h2, w_out


for _ in range(250):
    w_h1, w_h2, w_out = BackwardProp(y, x, w_h1, w_h2, w_out)
    _, _, xor = ForwardProp(x, w_h1, w_h2, w_out)
    print(xor)
    print(w_h1, w_h2, w_out)
