import numpy as np
from .functional import *
from .function import *

class Sigmoid(Function): #inherits Function Class 
						 #has attribute (dict) grad 
    def forward(self, X):
        """
        :Description: get forward path of sigmoid activation fn layer
        :parameter x:input to the layer.
        :type x: numpy array
        :return:sigmoid of input x
        """
        return sigmoid(X)

    def backward(self, dY):
        """
        :Description: get backward path of sigmoid activation fn layer
        :parameter dy:the product of partial derivative till this layer.
        :type dy: numpy array
        :return:dy * grad at x 
        """
        return dY * self.grad['X']

    def local_grad(self, X):
        grads = {'X': sigmoid_prime(X)} #dictionary 
        return grads

class ReLU(Function):
    def forward(self, X):
        """
        :Description: get forward path of RelU activation fn layer
        :parameter x:input to the layer.
        :type x: numpy array
        :return:RelU of input x
        """
        return relu(X)

    def backward(self, dY):
        """
        :Description: get backward path of Relu activation fn layer
        :parameter dy:the product of partial derivative till this layer.
        :type dy: numpy array
        :return:dy * grad at x 
        """
        return dY * self.grad['X']
        
    def local_grad(self, X):
        grads = {'X': relu_prime(X)}
        return grads

class LeakyReLU(Function):
    def __init__(self):
        super().__init__()
        self.alpha =alpha		
		
    def forward(self, X):
        """
        :Description: get forward path of LeakyRelu activation fn layer
        :parameter x:input to the layer.
        :type x: numpy array
        :return:LeakyRelu of input x
        """
        return leaky_relu(X,self.alpha)

    def backward(self, dY):
        """
        :Description: get backward path of LeakyRelu activation fn layer
        :parameter dy:the product of partial derivative till this layer.
        :type dy: numpy array
        :return:dy * grad at x 
        """
        return dY * self.grad['X']

    def local_grad(self, X):
        grads = {'X': leaky_relu_prime(X,self.alpha)}
        return grads

class Softmax(Function):

    def forward(self, X):
        """
        :Description: get forward path of Softmax activation fn layer
        :parameter x:input to the layer.
        :type x: numpy array
        :return:Softmax of input x
        """
        return softmax(X)

    def backward(self, dY):
        """
        :Description: get backward path of Softmax activation fn layer
        :parameter dy:the product of partial derivative till this layer.
        :type dy: numpy array
        :return:dy * 1
        """
        return dY 

    def local_grad(self,X):  
        pass 