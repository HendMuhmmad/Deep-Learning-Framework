import numpy as np
from math import sqrt
from itertools import product

from .layer import *
class Linear(Layer): 

  def __init__(self, in_dim, out_dim,optimizer_obj):
    super().__init__() 
    self._init_weights(in_dim, out_dim)
    self.optimizer = optimizer_obj
    super().set_optimizer(self.optimizer,isBatchNorm=False,isLinear=True)


  def _init_weights(self, in_dim, out_dim):

    """
    :Description:Gaussian distribution  initialization of weights
    :Parameter in_dim:dimesions of input to layer
    :Parameter ouy_dim:dimensions of output of layer
    :Returns:None
    """ 
    scale = 1 / sqrt(in_dim)
    self.weight['W'] = scale * np.random.randn(in_dim, out_dim) 
    self.weight['b'] = scale * np.random.randn(1, out_dim)	


  def forward(self, X):
      """
      :Description:Forward pass for the Linear layer.
      :Parameter X: numpy.ndarray of shape (n_batch, in_dim) containing
              the input value.

      :Returns:
          Y: numpy.ndarray of shape of shape (n_batch, out_dim) containing
              the output value.
      """
      output = np.dot(X, self.weight['W']) + self.weight['b']
      # caching variables for backprop (input : feature vector , output : score)
      self.cache['X'] = X 
      self.cache['output'] = output 
      return output

  def backward(self, dY):
      """
      :Description:Backward pass for the Linear layer.
      :Parameter dY: numpy.ndarray of shape (n_batch, n_out). Global gradient 
              backpropagated from the next layer.

      :Returns:
          dX: numpy.ndarray of shape (n_batch, n_out). Global gradient
              of the Linear layer.
      """
      dX = dY.dot(self.grad['X'].T) 

      # calculating the global gradient wrt to weights
      X = self.cache['X'] #input sample
      dW = self.grad['W'].T.dot(dY) 
      db = np.sum(dY, axis=0, keepdims=True) 

      
      # caching the global gradients
      self.weight_update = {'W': dW, 'b': db}
      return dX

  def local_grad(self, X):
      """
      :Description:Local gradients of the Linear layer at X.
      :Parameter X: numpy.ndarray of shape (n_batch, in_dim) containing the
              input data.

      :Returns: grads: dictionary of local gradients with the following items:
              X: numpy.ndarray of shape (n_batch, in_dim).
              W: numpy.ndarray of shape (n_batch, in_dim).
              b: numpy.ndarray of shape (n_batch, 1).
      """
      gradX_local = self.weight['W']
      gradW_local = X
      gradb_local = np.ones_like(self.weight['b'])
      grads = {'X': gradX_local, 'W': gradW_local, 'b': gradb_local}
      return grads