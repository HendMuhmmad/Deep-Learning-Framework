import numpy as np

class Evaluation:

  def __init__(self):
    pass

  def accuracy(self,y_true, y_pred):
    """
    :Description: 
    Calculate Accuracy of dataset.
    :parameter y_true: list of actual values
    :type y_true:array 
	parameter y_pred: list of predicted values
    :type y_pred:array
	:returns: correct_labels / len(y_true)
	""" 
    correct_labels = 0
    for true, pred in zip(y_true,y_pred):
      if true == pred:
        correct_labels += 1
    return correct_labels / len(y_true)
  
  def true_positive(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate number of true positives.
	:parameter positive:class label.
    :type positive: array. 
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: tp_counts.
	""" 
    tp_counts = 0
    for true, pred in zip(y_true, y_pred):
      if true==positive and pred==positive:
        tp_counts += 1
    return tp_counts
def true_negative(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate number of true negatives.
	:parameter positive:class label.
    :type positive: array. 
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: tn_counts.
	"""
    tn_counts = 0
    for true, pred in zip(y_true, y_pred):
      if true!=positive and pred!=positive:
        tn_counts += 1
    return tn_counts


def false_positive(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate number of false positives.
	:parameter positive:class label.
    :type positive: array. 
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: fp_counts.
	"""
    fp_counts = 0
    for true,pred in zip(y_true, y_pred):
      if true!=positive and pred==positive:
        fp_counts += 1
    return fp_counts

def false_negative(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate number of false negatives.
	:parameter positive:class label.
    :type positive: array. 
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: fn_counts.
	"""
    fn_counts = 0
    for true, pred in zip(y_true, y_pred):
      if true==positive and pred!=positive:
        fn_counts += 1
    return fn_counts

def precision_score(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate precision_score .
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: precision_value.
	""" 
    tp = self.true_positive(positive,y_true, y_pred)
    fp = self.false_positive(positive,y_true, y_pred)

    precision_value = tp/(tp+fp+1e-16)
    return precision_value

def recall_score(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate recall_score .
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: recall_value.
	"""
    tp = self.true_positive(positive,y_true, y_pred)
    fn = self.false_negative(positive,y_true, y_pred)
    recall_value = tp/(tp+fn+1e-16)
    return recall_value

def f1_score(self,positive,y_true, y_pred):
    """
    :Description: 
    Calculate f1_score .
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: f1_score_value.
	"""
    p = self.precision_score(positive,y_true, y_pred)
    r = self.recall_score(positive,y_true, y_pred)

    f1_score_value = 2 * p * r /(p+r+1e-16)
    return f1_score_value

def get_confusion_matrix_components(self,no_of_classes,y_ture, y_pred):
    """
    :Description: 
    Calculate confusion_matrix.
    :parameter no_of_classes: list of number of classes.
    :type no_of_classes: array .
    :parameter y_true: list of actual values.
    :type y_true: array .
	parameter y_pred: list of predicted values.
    :type y_pred: array.
	:returns: matrix dictionary.
	""" 

    array = np.empty((0),int)
    matrix_dict = {}
    true_class_list = list(range(1, no_of_classes + 1 ))
    predicted_class_list = list(range(1, no_of_classes + 1 ))
    for true_class in true_class_list :
      for predicted_class in predicted_class_list:
        cnt = 0
        for i in range(y_ture.shape[0]):
          if y_ture[i] == true_class and y_pred[i] == predicted_class :
            cnt +=1
        array = np.append(array,[cnt],axis=0)
    array = array.reshape(no_of_classes,-1)
    for i in range(no_of_classes) :
      matrix_dict["class_"+str(i+1)] = array[i]
      
    
    return matrix_dict