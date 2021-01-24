import numpy as np

class Model:

    __slots__ = ['layers', 'loss_fn','epochs_no']

    def __init__(self, layers, loss, epochs):
        assert isinstance(loss, Loss), 'loss must be an instance of nn.losses.Loss'
        assert isinstance(epochs,int), 'epochs must be an instance of int'
        for layer in layers:
            assert isinstance(layer, Function), 'layer should be an instance of ' \
                                                'nn.layers.Function or nn.layers.Layer'

        self.layers = layers
        self.loss_fn = loss
        self.epochs_no = epochs
	
    def __call__(self, *args, **kwargs):
      return self.forward(*args, **kwargs)
	
	# returns prediction for evaluation module to evaluate the accuracy 
	# returns losses for visualization module 
	# The dimensions of losses array  (no_of_epochs x 1) "column-vector"
  # The dimensions of losses array  ((no_of_epochs*N_samples) x 1) "column-vector"
  # because each sample has its own predicition and all the process is recalculates at each epoch
   
	
    def train(self,X_train,Y_train,lr): 
     

      
      losses_np = np.empty((0))
      pred_np = np.empty((0))

      print("start training ...")

      for epoch_idx in range(self.epochs_no):

        print("Epoch no. %d" % epoch_idx)
        out = self.__call__(X_train)
        
        print("call done ...")
        print("out shape : ",out.shape)
        
        # print("out : ",out)
        pred = np.argmax(out,axis=1)
        # print("pred-------------",pred.shape)

        # print("pred[0] =",pred[0])
        # print("pred shape : ",pred.shape)
        # pred = pred.reshape(pred.shape[0],1)
        # print("pred shape : ",pred.shape)
        # print("type pred =",type(pred))
        # print("pred = ",pred)
        pred_np = np.append(pred_np,pred,axis=0)
        
        
        print("prediction done ...")


        """  good calculation for accuracy 
        #accuracy = (1 - ( np.abs(pred - Y_train).sum()/(2*n_class_size) )) 
        #accuracy_np = np.append(accuracy_np, [accuracy], axis=0)
        #print("accuracy: %1.4f" % accuracy)
        """

        #total loss 
        loss = self.loss(out, Y_train)

        print("loss done ...")

        losses_np = np.append(losses_np, [loss], axis=0)
        #print('loss: %1.4f' % loss)

        #optimization technique
        grad = self.backward()
        self.update_weights(lr)

      return pred_np,losses_np
		
    def forward(self,x):
      """
      Calculates the forward pass by propagating the input through the
      layers.

      Args:
        x: numpy.ndarray. Input of the net.

      Returns:
        output: numpy.ndarray. Output of the net.
      """
      for layer in self.layers:
        x = layer(x)
      return x
		
	
    def loss(self, x, y):
      """
      Calculates the loss of the forward pass output with respect to y.
      Should be called after forward pass.

      Args:
      x: numpy.ndarray. Output of the forward pass.
      y: numpy.ndarray. Ground truth.

      Returns:
      loss: numpy.float. Loss value.
      """
      print("start loss .....")
      loss = self.loss_fn(x, y)
      return loss
	
    def backward(self):
      """
      Complete backward pass for the net. Should be called after the forward
      pass and the loss are calculated.

      Returns:
      d: numpy.ndarray of shape matching the input during forward pass.
      """
      d = self.loss_fn.backward() # first time when back-propagation first layer we face is loss fn 
      for layer in reversed(self.layers): # reversed(self.layers) : reversed the layers to make the backpropagation
        d = layer.backward(d) 
      return d
		
		
    def update_weights(self, lr):
      """
      Updates the weights for all layers using the corresponding gradients
      computed during backpropagation.

      Args:
      lr: float. Learning rate.
      """
      for layer in self.layers:
        if isinstance(layer, Layer):
          layer._update_weights(lr)


