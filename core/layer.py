from function import *

class Layer(Function):
    """
    Abstract model of a neural network layer. In addition to Function, a Layer
    also has weights and gradients with respect to the weights.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.weight = {} 
        self.weight_update = {} 
        self.optimizer = None
        self.isBatchNorm = False
        self.isLinear = False
    

    def set_optimizer(self,optimizer_obj,isBatchNorm=False,isLinear=False):
        """
        :Description: Set optimizer obj for layer
        :parameter optimizer_obj:optimizer obj for layer.
        :Parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
        :Parameter is linear:boolean parameter to indicate is it is Linear or not
        
        :return: None 
        """
        self.optimizer = optimizer_obj
        self.isBatchNorm = isBatchNorm
        self.isLinear = isLinear

    def _init_weights(self, *args, **kwargs):
      """
      :Description: initialize weights of layers
      """
        pass

    def _update_weights(self,epoch_no):

        """
        :Description:Updates the weights using the corresponding _global_ gradients computed during
        backpropagation.
        """
        self.weight = self.optimizer.__call__(self.weight,self.weight_update,epoch_no,self.isBatchNorm,self.isLinear)
       