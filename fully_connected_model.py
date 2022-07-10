import torch
from torch import nn
import numpy as np


######################################################################################################################################
# Design and train a two-layer fully-connected neural network with sigmoid nonlinearity and
# softmax cross entropy loss. Assuming an input dimension of D=784, a hidden dimension of H, and perform classification over C classes.
#
# The architecture should be fullyconnected -> sigmoid -> fullyconnected -> softmax.
#
# The learnable parameters of the model are stored in the dictionary,
# 'params', that maps parameter names to numpy arrays.
 ######################################################################################################################################

def TwoLayerNet(input_size, hidden_size, output_size, weight_init_std=0.01):
    """
    Initialize the weights and biases of the two-layer net. Weights
    should be initialized from a Gaussian with standard deviation equal to
    weight_init_std, and biases should be initialized to zero. All weights and
    biases should be stored in the dictionary 'params', with first layer
    weights and biases using the keys 'W1' and 'b1' and second layer weights
    and biases using the keys 'W2' and 'b2'.
    """
    params = {'W1': np.random.normal(0, weight_init_std, (input_size, hidden_size)),
              'W2': np.random.normal(0, weight_init_std, (hidden_size, output_size)),
              'b1': np.zeros(hidden_size),
              'b2': np.zeros(output_size)}
    return params


def FC_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, D) and contains a minibatch of N
    examples, where each example x[i] has shape D and will be transformed to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output result of the forward pass, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    out = np.dot(x, w) + b
    cache = (x, w, b)
    return out, cache


def FC_backward(dout, cache):
    """
    Computes the backward pass for a fully-connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = x.T.dot(dout)
    dx = dout.dot(w.T)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# Here we will design the entire model, which outputs the NN's probabilities and gradients.
def Model(params, x, t):
    """
    Computes the backward pass for a fully-connected layer.
    Inputs:
    - params:  dictionary with first layer weights and biases using the keys 'W1' and 'b1' and second layer weights
    and biases using the keys 'W2' and 'b2'. each with dimensions corresponding its input and output dimensions.
    - x: Input data, of shape (N,D)
    - t:  A numpy array of shape (N,C) containing training labels, it is a one-hot array,
      with t[GT]=1 and t=0 elsewhere, where GT is the ground truth label ;
    Returns:
    - y: the output probabilities for the minibatch (at the end of the forward pass) of shape (N,C)
    - grads: dictionary containing gradients of the loss with respect to W1, W2, b1, b2.
    """
    W1, W2 = params['W1'], params['W2']
    b1, b2 = params['b1'], params['b2']
    grads = {'W1': None, 'W2': None, 'b1': None, 'b2': None}
    batch_num = x.shape[0]

    # forward (fullyconnected -> sigmoid -> fullyconnected -> softmax).
    l1, c1 = FC_forward(x, W1, b1)
    s1 = sigmoid(l1)
    l2, c2 = FC_forward(s1, W2, b2)
    y = softmax(l2)

    # backward - calculate gradients.
    dy = (y - t) / batch_num
    dx, grads['W2'], grads['b2'] = FC_backward(dy, c2)

    dl1 = sigmoid_grad(l1) * dx
    _, grads['W1'], grads['b1'] = FC_backward(dl1, c1)
    return grads, y


def sigmoid(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig


def sigmoid_grad(x):
    s = sigmoid(x)
    sig_grad = s * (1 - s)
    return sig_grad


def softmax(x):
    """
  Inputs:
  - X: A numpy array of shape (N, C) containing a minibatch of data.
  Returns:
  - probabilities: A numpy array of shape (N, C) containing the softmax probabilities.
  if you are not careful here, it is easy to run into numeric instability
     """
    x -= np.max(x, axis=1, keepdims=True)
    probabilities = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return probabilities


def cross_entropy_error(y, t):
    """
    Inputs:
    - t:  A numpy array of shape (N,C) containing  a minibatch of training labels, it is a one-hot array,
      with t[GT]=1 and t=0 elsewhere, where GT is the ground truth label ;
    - y: A numpy array of shape (N, C) containing the softmax probabilities (the NN's output).
    Returns a tuple of:
    - loss as single float (do not forget to divide by the number of samples in the minibatch (N))
    """
    # Compute loss
    gt_idx = t.argmax(axis=1)
    batch_size = y.shape[0]
    n = np.arange(batch_size)
    error = tuple([-np.sum(np.log(y[n, gt_idx] + 1e-7)) / batch_size])  # Add 1e-7 prevent log(0)
    return error


 ######################################################################################################################################
 ##################################################### PyTorch Implementation #########################################################
 ######################################################################################################################################

class TwoLayerFC(nn.Module):
    def __init__(self, input_size=784, hidden_size=50, output_size=10):
        super(TwoLayerFC, self).__init__()
        self.fullyconnected1 = nn.Linear(input_size, hidden_size)
        self.fullyconnected2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fullyconnected1(x)
        x = torch.sigmoid(x)
        x = self.fullyconnected2(x)
        # x = self.softmax(x)  # In comment due to nn.CrossEntropyLoss
        return x