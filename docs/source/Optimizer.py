import random
import numpy as np

class Optimizer:

  def __init__(self,name ='', beta = 0, alpha = 0.1,raw = 0.999, raw_f = 0.9):

    self.name = name
    self.beta = beta
    self.alpha = alpha
    self.raw = raw
    self.raw_f = raw_f
    self.Ai = np.empty((0))
    self.Fi = np.empty((0))
    self.sigma = np.empty((0))
    self.momentum = np.empty((0))
    self.eps = 1e-6
    self.init = False
  
  def init_parameters(self,weight,isBatchNorm,isLinear):
    """
    :Description:initialize parameters for optimization algorithms 
    :Parameter weight: dictionary of the weight numpyarray & bias numpyarray  
    :Parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
    :Parameter is linear:boolean parameter to indicate is it is Linear or not

    :Returns: None
    """

    self.init = True
    if isLinear == True :#linear

      dimn0_w = weight['W'].shape[0] +1
      dimn1_w = weight['W'].shape[1]
      self.Ai = np.zeros((dimn0_w ,dimn1_w))
      self.Fi = np.zeros((dimn0_w ,dimn1_w))
      self.momentum  = np.zeros((dimn0_w ,dimn1_w))
      self.sigma = np.random.rand(dimn0_w ,dimn1_w)

    elif isLinear == False and isBatchNorm ==False:#conv
      dimn0_w = weight['W'].shape[0] 
      dimn1_w = weight['W'].shape[1]
      dimn2_w = weight['W'].shape[2]
      dimn3_w = weight['W'].shape[3]
      self.Ai = np.zeros((dimn0_w ,dimn1_w*dimn2_w*dimn3_w+1))
      self.Fi = np.zeros((dimn0_w ,dimn1_w*dimn2_w*dimn3_w+1))
      self.momentum  = np.zeros((dimn0_w ,dimn1_w*dimn2_w*dimn3_w+1))
      self.sigma = np.random.rand(dimn0_w ,dimn1_w*dimn2_w*dimn3_w+1)

    else : #batch normalization
      dimn0_gamma = weight['gamma'].shape[0]
      dimn1_gamma = weight['gamma'].shape[3]
      self.Ai_gamma = np.zeros((dimn0_gamma ,dimn1_gamma))
      self.Fi_gamma  = np.zeros((dimn0_gamma ,dimn1_gamma))
      self.momentum_gamma   = np.zeros((dimn0_gamma ,dimn1_gamma))
      self.sigma_gamma  = np.random.rand(dimn0_gamma ,dimn1_gamma)

      dimn0_beta = weight['beta'].shape[2]
      dimn1_beta = weight['beta'].shape[3]
      self.Ai_beta = np.zeros((dimn0_beta ,dimn1_beta))
      self.Fi_beta  = np.zeros((dimn0_beta ,dimn1_beta))
      self.momentum_beta   = np.zeros((dimn0_beta ,dimn1_beta))
      self.sigma_beta  = np.random.rand(dimn0_beta ,dimn1_beta)


  def __call__(self,weight,weight_update,epoch_no,isBatchNorm,isLinear):
    
    
    if self.init == False:
      self.init_parameters(weight,isBatchNorm,isLinear)

    if self.name == 'Momentum' :
      return self.update_parameters_with_momentum(weight,weight_update,epoch_no,isBatchNorm,isLinear)

    elif self.name == 'AdaDelta':
      return self.update_parameters_with_adaDelta(weight,weight_update,isBatchNorm,isLinear)

    elif self.name == 'AdaGrad':
      return self.update_parameters_with_adaGrad(weight,weight_update,isBatchNorm,isLinear)

    elif self.name == 'RMSProp':
      return self.update_parameters_with_RMSProp(weight,weight_update,isBatchNorm,isLinear)

    else :
      return self.update_parameters_with_GD(weight,weight_update,isBatchNorm,isLinear)
    

  def update_parameters_with_adaDelta(self,weight,weight_update,isBatchNorm,isLinear):
    """
    :Description:update weights with adaDelta optimization algorithms 
    :Parameter weight: dictionary of the weight numpyarray & bias numpyarray
    :Parameter weight_update:dictionary of the gradient numpyarray & bias numpyarray
    :Parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
    :Parameter islinear:boolean parameter to indicate is it is Linear or not

    :Returns: dictionary of the weight after updated numpyarray & bias numpyarray
    """

    if isBatchNorm == False :
      if isLinear == True :
        g = np.concatenate((weight_update['W'],weight_update['b']),axis=0)
        self.Ai = self.raw * self.Ai + (1.0 - self.raw) * g ** 2
        eta = np.sqrt((self.sigma + self.eps)/(self.Ai + self.eps))
        delta = - eta * g
        self.sigma = self.raw * self.sigma + (1.0 - self.raw) * delta ** 2
        weight['W'] = weight['W'] + delta[0:-1]
        weight['b'] = weight['b'] + delta[-1]
        # print("norm of gradient : " ,np.linalg.norm(g))
      else :
        dimn0 =  weight_update['W'].shape[0]
        dimn1 =  weight_update['W'].shape[1]
        dimn2 =  weight_update['W'].shape[2]
        dimn3 =  weight_update['W'].shape[3]

        weight_update_converted =  weight_update['W'].reshape(dimn0,-1)
        g = np.concatenate((weight_update_converted,weight_update['b']),axis=1)
        self.Ai = self.raw * self.Ai + (1.0 - self.raw) * g ** 2
        eta = np.sqrt((self.sigma + self.eps)/(self.Ai + self.eps))
        delta = - eta * g
        self.sigma = self.raw * self.sigma + (1.0 - self.raw) * delta ** 2

        weight_ = g[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        delta_W = delta[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        weight['W'] = weight_ + delta_W

        bias = g[:,dimn2*dimn3].reshape(-1,1)
        delta_b = (delta[:,dimn2*dimn3]).reshape(-1,1)
        weight['b'] = bias + delta_b

    else :
      #gamma
      self.Ai_gamma = self.raw * self.Ai_gamma + (1.0 - self.raw) * weight_update['gamma'] ** 2
      eta_gamma = np.sqrt((self.sigma_gamma + self.eps)/(self.Ai_gamma + self.eps))
      delta_gamma = - eta_gamma * weight_update['gamma']
      self.sigma_gamma = self.raw * self.sigma_gamma + (1.0 - self.raw) * delta_gamma ** 2
      weight['gamma'] = weight['gamma'] + delta_gamma

      #beta
      self.Ai_beta = self.raw * self.Ai_beta + (1.0 - self.raw) * weight_update['beta'] ** 2
      eta_beta = np.sqrt((self.sigma_beta + self.eps)/(self.Ai_beta + self.eps))
      delta_beta = - eta_beta * weight_update['beta']
      self.sigma_beta = self.raw * self.sigma_beta + (1.0 - self.raw) * delta_beta ** 2
      weight['beta'] = weight['beta'] + delta_beta

    return weight


  def update_parameters_with_adaGrad(self,weight,weight_update,isBatchNorm,isLinear):
    """
    :Description:update weights with adaGrad optimization algorithms 
    :parameter weight: dictionary of the weight numpyarray & bias numpyarray
    :parameter weight_update:dictionary of the gradient numpyarray & bias numpyarray
    :parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
    :parameter islinear:boolean parameter to indicate is it is Linear or not
    :returns: dictionary of the weight after updated numpyarray & bias numpyarray
    """      
    if isBatchNorm == False:
      if isLinear == True :
        g = np.concatenate((weight_update['W'],weight_update['b']),axis=0)
        self.Ai = self.Ai + g ** 2
        eta = self.alpha/np.sqrt((self.Ai + self.eps))
        delta = - eta * g
        weight['W'] = weight['W'] + delta[0:-1]
        weight['b'] = weight['b'] + delta[-1]

      else :

        dimn0 =  weight_update['W'].shape[0]
        dimn1 =  weight_update['W'].shape[1]
        dimn2 =  weight_update['W'].shape[2]
        dimn3 =  weight_update['W'].shape[3]

        weight_update_converted =  weight_update['W'].reshape(dimn0,-1)
        g = np.concatenate((weight_update_converted,weight_update['b']),axis=1)
        self.Ai = self.Ai + g ** 2
        eta = self.alpha/np.sqrt((self.Ai + self.eps))
        delta = - eta * g

        weight_ = g[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        delta_W = delta[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        weight['W'] = weight_ + delta_W

        bias = g[:,dimn2*dimn3].reshape(-1,1)
        delta_b = (delta[:,dimn2*dimn3]).reshape(-1,1)
        weight['b'] = bias + delta_b



    else : #batch normalization
      #gamma
      self.Ai_gamma = self.Ai_gamma + weight_update['gamma'] ** 2
      eta_gamma = self.alpha/np.sqrt((self.Ai_gamma + self.eps))
      delta_gamma = - eta_gamma * weight_update['gamma']
      weight['gamma'] = weight['gamma'] + delta_gamma

      #beta
      self.Ai_beta = self.Ai_beta + weight_update['beta'] ** 2
      eta_beta = self.alpha/np.sqrt((self.Ai_beta + self.eps))
      delta_beta = - eta_beta * weight_update['beta']
      weight['beta'] = weight['beta'] + delta_beta

    return weight



  def update_parameters_with_RMSProp(self,weight,weight_update,isBatchNorm,isLinear):
    """
    :Description:update weights with RMSProp optimization algorithms 
    :Parameter weight: dictionary of the weight numpyarray & bias numpyarray
    :Parameter weight_update:dictionary of the gradient numpyarray & bias numpyarray
    :Parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
    :Parameter islinear:boolean parameter to indicate is it is Linear or not

    :Returns: dictionary of the weight after updated numpyarray & bias numpyarray
    """
    if isBatchNorm == False:
      if isLinear == True : #linear
        g = np.concatenate((weight_update['W'],weight_update['b']),axis=0)
        self.Ai = self.raw * self.Ai + (1.0 - self.raw) * g ** 2
        eta = self.alpha/np.sqrt((self.Ai + self.eps))
        delta = - eta * g
        weight['W'] = weight['W'] + delta[0:-1]
        weight['b'] = weight['b'] + delta[-1]

      else :#conv
        dimn0 =  weight_update['W'].shape[0]
        dimn1 =  weight_update['W'].shape[1]
        dimn2 =  weight_update['W'].shape[2]
        dimn3 =  weight_update['W'].shape[3]

        weight_update_converted =  weight_update['W'].reshape(dimn0,-1)
        g = np.concatenate((weight_update_converted,weight_update['b']),axis=1)
        self.Ai = self.raw * self.Ai + (1.0 - self.raw) * g ** 2
        eta = self.alpha/np.sqrt((self.Ai + self.eps))
        delta = - eta * g

        weight_ = g[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        delta_W = delta[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        weight['W'] = weight_ + delta_W

        bias = g[:,dimn2*dimn3].reshape(-1,1)
        delta_b = (delta[:,dimn2*dimn3]).reshape(-1,1)
        weight['b'] = bias + delta_b


    else :#batchNorm

      #gamma
      self.Ai_gamma = self.raw * self.Ai_gamma + (1.0 - self.raw) * weight_update['gamma'] ** 2
      eta_gamma = self.alpha/np.sqrt((self.Ai_gamma + self.eps))
      delta_gamma = - eta_gamma * weight_update['gamma']
      weight['gamma'] = weight['gamma'] + delta_gamma

      #beta
      self.Ai_beta = self.raw * self.Ai_beta + (1.0 - self.raw) * weight_update['beta'] ** 2
      eta_beta = self.alpha/np.sqrt((self.Ai_beta + self.eps))
      delta_beta = - eta_beta * weight_update['beta']
      weight['beta'] = weight['beta'] + delta_beta

       
    return weight


  def update_parameters_with_momentum(self,weight,weight_update,epoch_no,isBatchNorm,isLinear):
    """
      :Description:update weights with Momentum optimization algorithms 
      :Parameter weight: dictionary of the weight numpyarray & bias numpyarray
      :Parameter weight_update:dictionary of the gradient numpyarray & bias numpyarray
      :Parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
      :Parameter islinear:boolean parameter to indicate is it is Linear or not

      :Returns: dictionary of the weight after updated numpyarray & bias numpyarray
      """
    if isBatchNorm == False :
      if isLinear == True : #linear
        print("epoch at momentum :",epoch_no)
        g = np.concatenate((weight_update['W'],weight_update['b']),axis=0)
        delta =  - self.alpha * g
        self.momentum = self.beta * self.momentum + delta
        weight['W'] = weight['W'] + self.momentum[0:-1]
        weight['b'] = weight['b'] + self.momentum[-1]

      else : #conv
        dimn0 =  weight_update['W'].shape[0]
        dimn1 =  weight_update['W'].shape[1]
        dimn2 =  weight_update['W'].shape[2]
        dimn3 =  weight_update['W'].shape[3]

        weight_update_converted =  weight_update['W'].reshape(dimn0,-1)
        g = np.concatenate((weight_update_converted,weight_update['b']),axis=1)
        delta =  - self.alpha * g
        self.momentum = self.beta * self.momentum + delta

        weight_ = g[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        momentum_W = self.momentum[:,0:-1].reshape(dimn0,dimn1,dimn2,dimn3)
        weight['W'] = weight_ + momentum_W

        bias = g[:,dimn2*dimn3].reshape(-1,1)
        momentum_b = (self.momentum[:,dimn2*dimn3]).reshape(-1,1)
        weight['b'] = bias + momentum_b

    else :
      #gamma
      delta_gamma =  - self.alpha * weight_update['gamma']
      self.momentum_gamma = self.beta * self.momentum_gamma + delta_gamma
      weight['gamma'] = weight['gamma'] + self.momentum_gamma

      #beta
      delta_beta =  - self.alpha * weight_update['beta']
      self.momentum_beta = self.beta * self.momentum_beta + delta_beta
      weight['beta'] = weight['beta'] + self.momentum_beta
      
    return weight

  
  def update_parameters_with_adam(self,weight,weight_update):

    """
      :Description:update weights with Adam optmization algorithm
      :Parameter weight: dictionary of the weight numpyarray & bias numpyarray
      :Parameter weight_update:dictionary of the gradient numpyarray & bias numpyarray
      :Returns: dictionary of the weight after updated numpyarray & bias numpyarray
      """
    g = np.concatenate((weight_update['W'],weight_update['b']),axis=0)
    self.Ai = self.raw * self.Ai + (1.0 - self.raw) * g ** 2
    self.Fi = self.raw_f * self.Fi + (1.0 - self.raw_f) * g
    Fi_hat = self.Fi / (1.0 - self.raw_f ** self.epoch_no)
    Ai_hat = self.Ai / (1.0 - self.raw ** self.epoch_no)
    eta = self.alpha / (np.sqrt(Ai_hat) + self.eps)
    weight['W'] = weight['W'] + eta * Fi_hat[0:-1]
    weight['b'] = weight['b'] + eta * Fi_hat[-1]
    return weight

  def update_parameters_with_GD(self,weight,weight_update,isBatchNorm,isLinear):
    """
    :Description:update weights with Normal Gradient descent 
    :Parameter weight: dictionary of the weight numpyarray & bias numpyarray
    :Parameter weight_update:dictionary of the gradient numpyarray & bias numpyarray
    :Parameter isBatchNorm:boolean parameter to indicate is it batch normalization or not
    :Parameter islinear:boolean parameter to indicate is it is Linear or not

    :Returns: dictionary of the weight after updated numpyarray & bias numpyarray
    """
    for weight_key,_ in weight.items():
      weight[weight_key] = weight[weight_key] - self.alpha * weight_update[weight_key]
    return weight