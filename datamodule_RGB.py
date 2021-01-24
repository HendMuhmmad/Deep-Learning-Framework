from numpy.random import RandomState
import random
from numpy import asarray
from PIL import Image
import pandas as pd
import numpy as np 
import collections
import os
from zipfile import ZipFile
from pathlib import Path
import shutil
import py7zr 
import win32file
from os import listdir
from PIL import Image as PImage
import os.path


def load_data_RGB():
  """
  :Description:

  downloads CIFAR-10 dataset from kaggle.
  the files are "train.7z", "test.7z" and "trainLabels.csv"
  creates a folder called "Train_validation" and move "train.7z" to it 
  the files are in 7z format so this function also unzip the file "train.7z".

  :returns: none.

  """
  os.system('kaggle competitions download -c "cifar-10"')
  zip = ZipFile('cifar-10.zip')
  zip.extractall()
  os.system('mkdir Train_validation')
  shutil.move('train.7z', './Train_validation')
  os.chdir('Train_validation')
  with py7zr.SevenZipFile('train.7z', mode='r') as z: 
    z.extractall()

def load_test():
  """
  :Description:
  
  creates a new Test folder and moves the test images to it.

    
  :IMPORTANT:

  make sure you are on the right path (we suppose you are in the main folder)
  "if you run load_data_RGB() first, you have to cd to the main folder "

  :returns: none.

  """
  os.system('mkdir Test')
  shutil.move('test.7z', './Test')
  os.chdir('Test')
  with py7zr.SevenZipFile('test.7z', mode='r') as z: 
    z.extractall()

def process_data_RGB():
  """
  :Description:

  makes a dictionary that includes all photos in "train" file then sorting them
  according to the name of the .png file
  the key of the dictionary is the name of the .png file after we convert
  it to a number like 1 , 2 , 3 , 4 .... etc
  and the value of the dictionary is the image itself.
  then we sort this dictionary according to the key numbers

  :IMPORTANT:

  make sure you are on the right path (we suppose you are in Train_validation after running load_data_RGB())

  :returns: The Dictionary of the sorted image.

  """
  Images_Dict = {}
  Pixels_List = []
  os.chdir('train')   
  Images_Dict = {}
  imagesList = listdir('./')
  loadedImages = []
  for image in imagesList:
    with open(os.path.join('./', image), 'rb') as i:
      img = PImage.open(i)
      updated_name =i.name.replace(i.name[0]+i.name[1], '')
      first_part = updated_name.split('.')
      number = int(first_part[0])
      loadedImages.append(img)
      pixel = asarray(img)
      Pixels_List.append(pixel)
      Images_Dict[number] = pixel
  Ordered_images_Dict = collections.OrderedDict(sorted(Images_Dict.items()))
  return Ordered_images_Dict
  
def get_pixels_array_RGB(Dict):
  """
  :Description:

  it takes the images from the ordered dictionary and append them to a list
  called "Image_List"
  then convert the images in "Image_List" to pixels using "asarray()" 
  and append them to a list (Pixels_List)
  we read the "trainLabels.csv" file and convert it to a list (Labels_List)
  the we randomly split "Pixels_List" to training and validation with a ratio
  of 80% training and 20% validation
  we do the same thing for Labels
  then we convert training and validation Lists to numpy arrays.

  :parameter images_dict: input dictionary of train images.
  :type images_dict: dictionary of 50000 images.

  :IMPORTANT:

  make sure you are on the right path (we suppose you are in train after running process_data_RGB())

  :returns: a numpy arrays of the training data (40000x32x32x3), validation data (10000x32x32x3), training labels (40000x1) and
  validation labels (10000x1)

  """
  Ordered_images_Dict = Dict 
  Pixels_List = []
  train = []
  train_labels = []
  validation_Label_List = []
  train_label_List = []

  for key, value in Ordered_images_Dict.items():
    Pixels_List.append(value)

  labels = pd.read_csv("../../trainLabels.csv") 
  Labels_List = labels.values.tolist()
  idx = random.sample(range(50000), 40000)
  for i in idx:
    train.append(Pixels_List[i])
    train_labels.append(Labels_List[i])
  validation_List = [i for j, i in enumerate(Pixels_List) if j not in idx]
  validation_Label = [i for j, i in enumerate(Labels_List) if j not in idx]
  validation_array = np.asarray(validation_List)
  Pixels_array = np.asarray(train)
  for i in train_labels:
    train_label_List.append(i[1]) 
  for i in validation_Label:
    validation_Label_List.append(i[1])
  validation_Label_array = np.asarray(validation_Label_List)
  trian_Label_array = np.asarray(train_label_List)
  print(Pixels_List[0])
  print(type(Pixels_List[0]))
  print(Pixels_List[0].shape)
  return Pixels_array,validation_array,trian_Label_array,validation_Label_array

def get_test_pixels():
  """
  :Description:
  creates a numpy array containing the pixels of the images.
  
  :IMPORTANT:

  make sure you are on the right path (we suppose you are in the Test folder after running load_test() )


  :returns: numpy array of the pixels of the images(300000x32x32x3).

  """
  os.chdir('test') 
  Pixels_List = []  
  imagesList = listdir('./')
  for image in imagesList:
    with open(os.path.join('./', image), 'rb') as i:
      img = PImage.open(i)
      pixel = asarray(img)
      Pixels_List.append(pixel)
  Pixels_array = np.asarray(Pixels_List)
  return Pixels_array

# To try the functions
load_data_RGB()
Dict = process_data_RGB()
Pixels_array,validation_array,trian_Label_array,validation_Label_array=get_pixels_array_RGB(Dict)
# we don't recommend trying those lines :) (because the test has 300000 image it will take alot of time to load and unzip)
load_test()
test_Array = get_test_pixels()




