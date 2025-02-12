from numpy.random import RandomState
import pandas as pd
import os
import numpy
from zipfile import ZipFile
from pathlib import Path
import shutil

def load_data():
  """
  :Description:

  downloads the digit-recognizer data from kaggle as 2 zip files one contains
  the test.csv which has 28000 test image and the other contains 42000 train and
  validation images, then it unzips 
  the data into their specific folders after splitting the train and validation
  datasets.

  :returns: the train, validation and test datasets in dataframe format
  """
  os.system('kaggle competitions download -c "digit-recognizer"')
  zip = ZipFile('digit-recognizer.zip')
  zip.extractall()
  os.system('mkdir validation')
  os.system('mkdir train')
  os.system('mkdir labels')
  os.system('mkdir train_validation')
  os.system('mkdir test')
  shutil.move('test.csv', './test')
  shutil.move('train.csv', './train_validation')

  df = pd.read_csv('train_validation/train.csv')
  test = pd.read_csv('test/test.csv')
  #to generate the same data everytime we split
  train = df.sample(frac=0.8, random_state=8)
  validation = df.loc[~df.index.isin(train.index)]
  validation.to_csv('validation/validation.csv', index=False)
  train.to_csv('train/train.csv',index=False)
  train = train.reset_index(drop=True)
  validation = validation.reset_index(drop=True)
  return train,validation,test


def transform_data(train_,validation_,test_):
  """
  :Description:

  splits the features and the labels of the train and validation datasets
  then transforms the dataframes into numpy arrays.

  :parameter train_: input dataframe of train samples.
  :type train_: dataframe of 33600 images with their pixels, its shape (33600x785).

  :parameter validation_: input dataframe of validation samples.
  :type validation_: dataframe of 8400 images with their pixels, its shape (8400x785).

  :parameter test_: input dataframe of test samples.
  :type test_: dataframe of 28000 images with their pixels.

  :returns: 5 numpy arrays which are train_labels_array (33600x1), validation_labels_array (8400x1)
  train_array (33600x784), validation_array (8400x784), test_array (28000x784)
  """
  train = train_
  validation = validation_
  test = test_
  #train
  train_label = train['label']
  train_label.to_csv('labels/train_labels.csv',index=False)
  train_labels_array = numpy.loadtxt(open("labels/train_labels.csv", "rb"), delimiter=",")
  train = train.drop(columns=['label'])
  train.to_csv('train/train.csv',index=False )
  train_array = numpy.loadtxt(open("train/train.csv", "rb"), delimiter=",", skiprows=1)
  #validation
  validation_label = validation['label']
  validation_label.to_csv('labels/validation_labels.csv',index=False )
  validation_labels_array = numpy.loadtxt(open("labels/validation_labels.csv", "rb"), delimiter=",")
  validation = validation.drop(columns=['label'])
  validation.to_csv('validation/validation.csv',index=False )
  validation_array = numpy.loadtxt(open("validation/validation.csv", "rb"), delimiter=",", skiprows=1)  
  #test
  test_array = numpy.loadtxt(open("test/test.csv", "rb"), delimiter=",", skiprows=1)
  return train_labels_array,validation_labels_array,train_array,validation_array,test_array


def normalize_data(train_array_,validation_array_):
  """
  :Description:

  normalize the train and validation arrays by dividing pixels values by 255.

  :parameter train_array_: input numpy array of train samples.
  :type train_array_: numpy array of 33600 images with their pixels, its shape (33600x784).

  :parameter validation_array_: input numpy array of validation samples.
  :type validation_array_: numpy array of 8400 images with their pixels, its shape (8400x784).

  :returns: the train and validation numpy arrays after normalization
  """
  train_array = train_array_
  validation_array = validation_array_
  train_array = train_array.astype('float32')
  train_array /= 255
  validation_array = validation_array.astype('float32')
  validation_array /= 255
  return train_array,validation_array


# Try to run the functions:
train,validation,test = load_data()
train_labels_array,validation_labels_array,train_array,validation_array,test_array = transform_data(train,validation,test)
train_array,validation_array = normalize_data(train_array,validation_array)
