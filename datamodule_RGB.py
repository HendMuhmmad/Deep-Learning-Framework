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





