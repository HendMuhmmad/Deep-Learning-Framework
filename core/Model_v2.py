import numpy as np

class Model(Net):

	__slots__ = ['epochs_no'] # must add slot for optimizer name ['epochs_no','optimizer_type']
							  # second approach suggests optimizer_object not name 
	
	def __init__(self, layers, loss, epochs):#,optimizer
		super().__init__(layers, loss)
		assert isinstance(epochs,int),'epochs must be an instance of int'
		self.epochs_no = epochs
		#self.optimizer_type = optimizer
		
		
		
	def train(self,X_train,Y_train,lr): 
     

      
      losses_np = np.empty((0))
      pred_np = np.empty((0))

      print("start training ...")

      for epoch_idx in range(self.epochs_no):

        print("Epoch no. %d" % epoch_idx)
        out = super().__call__(X_train)
        
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
        loss = super().loss(out, Y_train)

        print("loss done ...")

        losses_np = np.append(losses_np, [loss], axis=0)
        #print('loss: %1.4f' % loss)

        #optimization technique
        grad = super().backward()
        super().update_weights(lr) # call to optimizer and pass needed 
								   # we will add optimizer_name(self.optimizer_type) with lr 
								   
								   # second approach suggest super().update_weights(optimizer_object)

      return pred_np,losses_np
		
    
	