import numpy as np
      
def sigmoid(x):
    """
    :Description:
    calculate sigmoid fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: sigmoid of x
    """
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    """
    :Description:
    calculate derivative sigmoid fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: derivative sigmoid of x
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """
    :Description:
    calculate relu fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: relu of x
    """
    return x*(x > 0)

def relu_prime(x):
    """
    :Description:
    calculate derivative relu fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: derivative relu of x
    """
    return 1*(x > 0)

def leaky_relu(x, alpha):
  """
    :Description:
    calculate leaky_relu fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: leaky_relu of x
    """
    return x*(x > 0) + alpha*x*(x <= 0)

def leaky_relu_prime(x, alpha):
     """
    :Description:
    calculate derivative leaky_relu fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: derivative leaky_relu of x
    """
    return 1*(x > 0) + alpha*(x <= 0)

def softmax(x):
    """
    :Description:
    calculate softmax fn
    :parameter x:required value for calculation.
    :type x: numpy array
     returns: softmax of x
    """
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return probs