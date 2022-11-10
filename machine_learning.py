###
# This file was created by Sara Millan (smillan@tamu.edu) for the Aggie Map Navigation Helper team for Senior Capstone Project.
# The Aggie Map Navigation Helper guides users through the Texas A&M University campus via an Android App.
# The user inputs their destination and a picture of their surroundings(with the nearest building)and the phone app will
# determine their location. This code completes the machine learning, image classification, and image similarity part of the
# backend. An image is taken from an S3 upload, classified, and then compared to other images in its class to locate the user.
# The coordinates of the images in its class are known and used to triangulate the user's location.
###

import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import image 
from PIL import Image  
import pandas as pd
from pandas import DataFrame
from keras.preprocessing import image
import boto3
from scipy import spatial
from numpy.core.fromnumeric import sort
import scipy.misc
from scipy import ndimage
import csv
import math

def image_class(event, context):  
  # -*- coding: utf-8 -*-

  # Upload model from within the container image
  model = keras.models.load_model('404-KCV/')

  # labels for different classes of images (building names with a side number)
  CLASS_NAMES= ['ZACH6', 'ZACH5', 'ZACH4', 'ZACH3', 'ZACH2', 'ZACH1', 'WEB4', 'WEB3', 'WEB2', 'WEB1', 'PETE3', 'PETE2', 'PETE1', 'MEOB3', 'MEOB2', 'MEOB1', 'HEB4', 'HEB3', 'HEB2', 'HEB1', 'ETB5', 'ETB4', 'ETB3', 'ETB2', 'ETB1', 'DLEB1', 'CVLB2', 'CVLB1', 'CHEN3', 'CHEN2', 'CHEN1']

  # ML Prediction

  # Retrieve image from S3 and classify the image
  imagepath = 'user-image.jpg' # Name of user image pushed from Android phone
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/user-image.jpg', '/tmp/user-image.jpg')
  image = Image.open('/tmp/user-image.jpg')
  small_image = image.resize((224,224)) # input size for VGG16 224,224
  small_imgarr = np.array(small_image)
  img = np.expand_dims(small_imgarr, axis=0)
  output = model.predict(img)

  # dataframe of all of the database coordinates from the container image
  df = pd.read_csv('FINAL_CSV.csv')


  # Cosine Similarity
  
  TestArray = img
  TestArray = TestArray.flatten()
  compArray = [] # comparison array of the user input image
  SimArray = [] # similarity percentage array
  SimHold = [] # array for holding images from individual classes as outputted by the classifier
  indexHold = [] # array for holding original image index from classes in SimHold

  # only does cosine distance between test image and images from database of its class
  for i in range(np.argmax(output)*1000, (np.argmax(output)+1)*1000): # total number of images in dataset
    imagepath = 'content/IMG_' + df['FileName'][i][4:10] + '.jpg'
    image = Image.open(imagepath)
    compArray = np.array(image)
    compArray = compArray.flatten()
    dist = 1 - spatial.distance.cosine(TestArray, compArray)
    SimArray.append(dist)
    indexHold.append(i)


  # Sort Array to get top 5 images

  # get indexes of these percentages to pull the images from the comparison set
  sort_index = np.argsort(SimArray)
  sort_index = sort_index[::-1]
  sort_index = (np.argmax(output)*1000) + sort_index

  tot_lat = 0
  tot_lon = 0
  for i in range(0,5):
    # get gps data from dataframe at same indexes as images
    tot_lat = tot_lat + float(df['GPSLatitude'][sort_index[i]])
    tot_lon = tot_lon + float(df['GPSLongitude'][sort_index[i]])

  # do once prediction is figured out
  avg_lat = tot_lat / 5
  avg_lon = tot_lon / 5


  # Pause the code while user is selecting destination
  # Read a csv file that will update a value to tell this function when to continue
  s3 = boto3.client('s3')
  csvfile = s3.get_object(Bucket='user-input-image', Key='public/dest-state.csv')
  csvcontent = csvfile['Body'].read().split(b'\n')
  # pause the code while the user is still selecting their destination (dest-state.csv first cell==0)
  while csvcontent == "0":
    csvfile = s3.get_object(Bucket='user-input-image', Key='public/dest-state.csv')
    csvcontent = csvfile['Body'].read().split(b'\n')


  # Opens the end coordinate csv to access the destination coordinates
  # This is used to create a new csv that is pushed to s3 with the start coordinate included with the end coordinate
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/end_coordinates.csv', '/tmp/end_coordinates.csv') 
  CSVData = open('/tmp/end_coordinates.csv')
  coord_arr = np.genfromtxt(CSVData, delimiter=",")
  coord_arr[0,0] = avg_lat;
  coord_arr[0,1] = avg_lon;
  # write the updated csv to s3
  temp_csv_file = csv.writer(open("/tmp/start_end_coordinates.csv", "w+"))
  np.savetxt('/tmp/start_end_coordinates.csv', coord_arr, delimiter=",")
  BUCKET_NAME = 'user-input-image'
  client = boto3.client('s3')
  client.upload_file('/tmp/start_end_coordinates.csv', BUCKET_NAME,'public/start_end_coordinates.csv')

  return avg_lat, avg_lon
