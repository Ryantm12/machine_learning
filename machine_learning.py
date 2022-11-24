###
# This file was created by the Aggie Map Navigation Helper team for Texas A&M Senior Capstone Project.
# The Aggie Map Navigation Helper guides users through the Texas A&M University campus via an Android App.
# The user inputs their destination and a picture of their surroundings(with the nearest building)and the phone app will
# determine their location. This code completes the machine learning, image classification, and image similarity part of the
# backend. An image is taken from an S3 upload, classified, and then compared to other images in its class to locate the user.
# The coordinates of the images in its class are known and used to triangulate the user's location.
# 
# After this, the pathfinder code guides the user through campus from their current location to their destination.
# The pathfinder code takes in a set of two gps coordinates, one for a user's current location, and another for destination.
# Dijkstra's algorithm is used to determine the shortest possible path. The result is output back to S3 for the app to read and display to the user.
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
from math import sin, cos, asin, sqrt, pi
import geopandas as gpd
from shapely.geometry import LineString, Point
import osmnx as ox
import networkx as nx
import os.path


def image_class(event, context):  
  # -*- coding: utf-8 -*-
  # Upload the models from within the container image
  binary_model = keras.models.load_model('404-binary2/')
  model = keras.models.load_model('404-KCV/')
  # labels for different classes of images (building names with a side number)
  CLASS_NAMES= ['ZACH6', 'ZACH5', 'ZACH4', 'ZACH3', 'ZACH2', 'ZACH1', 'WEB4', 'WEB3', 'WEB2', 'WEB1', 'PETE3', 'PETE2', 'PETE1', 'MEOB3', 'MEOB2', 'MEOB1', 'HEB4', 'HEB3', 'HEB2', 'HEB1', 'ETB5', 'ETB4', 'ETB3', 'ETB2', 'ETB1', 'DLEB1', 'CVLB2', 'CVLB1', 'CHEN3', 'CHEN2', 'CHEN1']
  # Pause the code while the user is taking an image
  # Read a csv file that will update a value to tell this code when to resume
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/image-state.csv', '/tmp/image-state.csv') 
  CSVData = open('/tmp/image-state.csv')
  csvcontent = np.genfromtxt(CSVData, delimiter=",")
  # pause the code while the user is still taking an image (image-state.csv first cell==0)
  while csvcontent == 0:
    s3.Bucket('user-input-image').download_file('public/image-state.csv', '/tmp/image-state.csv') 
    CSVData = open('/tmp/image-state.csv')
    csvcontent = np.genfromtxt(CSVData, delimiter=",")

  # ML Prediction
  # Retrieve image from S3 and classify the image
  imagepath = 'user-image.jpg' # Name of user image pushed from Android phone
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/user-image.jpg', '/tmp/user-image.jpg')
  image = Image.open('/tmp/user-image.jpg')
  small_image = image.resize((224,224)) # input size for VGG16 224,224
  small_imgarr = np.array(small_image)
  img = np.expand_dims(small_imgarr, axis=0)
  # First check to see if the image is a building or not. If it is, write a 0 to the binary.csv and continue.
  # If not, write a 1 to the binary.csv and terminate.
  binary_output = binary_model.predict(img)
  print(binary_output)
  BUCKET_NAME = 'user-input-image'
  if (binary_output[0][0] <  0.5):
    # Update a csv file(write a 0) in S3 to tell Android studio that the image was a building
    s3 = boto3.resource('s3')
    s3.Bucket('user-input-image').download_file('public/binary.csv', '/tmp/binary.csv') 
    CSVData = open('/tmp/binary.csv')
    state_var = [0]
    np.savetxt('/tmp/binary.csv', state_var, delimiter=",")
    client = boto3.client('s3')
    client.upload_file('/tmp/binary.csv', BUCKET_NAME,'public/binary.csv')
  else:
    # Update a csv file(write a 1) in S3 to tell Android studio that the image was not a building
    s3 = boto3.resource('s3')
    s3.Bucket('user-input-image').download_file('public/binary.csv', '/tmp/binary.csv') 
    CSVData = open('/tmp/binary.csv')
    state_var = [1]
    np.savetxt('/tmp/binary.csv', state_var, delimiter=",")
    client = boto3.client('s3')
    client.upload_file('/tmp/binary.csv', BUCKET_NAME,'public/binary.csv')
    #Then kill the runtime
    return 0
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
  # Get the indexes of these percentages to pull the top 5 images from the comparison set
  sort_index = np.argsort(SimArray)
  sort_index = sort_index[::-1]
  sort_index = (np.argmax(output)*1000) + sort_index
  tot_lat = 0
  tot_lon = 0
  for i in range(0,5):
    # get gps data from dataframe at same indexes as images
    tot_lat = tot_lat + float(df['GPSLatitude'][sort_index[i]])
    tot_lon = tot_lon + float(df['GPSLongitude'][sort_index[i]])
  # Average the summed latitude and longitude
  avg_lat = tot_lat / 5
  avg_lon = tot_lon / 5

  # Pause the code while user is selecting destination
  # Read a csv file that will update a value to tell this function when to continue
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/dest-state.csv', '/tmp/dest-state.csv') 
  CSVData = open('/tmp/dest-state.csv')
  csvcontent = np.genfromtxt(CSVData, delimiter=",")
  # pause the code while the user is still selecting their destination (dest-state.csv first cell==0)
  while csvcontent == 0:
    s3.Bucket('user-input-image').download_file('public/dest-state.csv', '/tmp/dest-state.csv') 
    CSVData = open('/tmp/dest-state.csv')
    csvcontent = np.genfromtxt(CSVData, delimiter=",")
  # Opens the end coordinate csv to access the destination coordinates
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/end_coordinates.csv', '/tmp/end_coordinates.csv') 
  CSVData = open('/tmp/end_coordinates.csv')
  coord_arr = np.genfromtxt(CSVData, delimiter=",")
  coord_arr[0,0] = avg_lat
  coord_arr[0,1] = avg_lon


  #      Pathfinder Code

  # This code finds the shortest distance path between two points(current and destination) on the network(graph)
  # This function performs the following actions in order:
  # 1. Convert input coordinates
  # 2. Find the shortest path
  # 3. Format results for output as a list
  # 4. Calculate shortest path length

  # Collect user current location and destination
  coord1 = coord_arr[0,0]
  coord2 = coord_arr[0,1]
  coord3 = coord_arr[1,0]
  coord4 = coord_arr[1,1]
  # Pull in graph from .graphml file
  graph_paths = ox.load_graphml(filepath='graph.graphml')
  # Create separate GeoDataFrames to store both locations
  # It takes in a numpy array and returns the two location GeoDataFrames
  # Include CRS 4326
  current_coord = gpd.GeoDataFrame(columns=['geometry'], crs=4326, geometry='geometry')
  current_coord.at[0, 'geometry'] = Point(coord2, coord1)
  destination_coord = gpd.GeoDataFrame(columns=['geometry'], crs=4326, geometry='geometry')
  destination_coord.at[0, 'geometry'] = Point(coord4, coord3)

  # Find the shortest path
  # Take array of dots and project them
  graph_proj = ox.project_graph(graph_paths)
  # Convert projection into a geo-dataframe
  map_edges = ox.graph_to_gdfs(graph_proj, nodes=False)
  # Returns the Coordinate Reference system of the GeoDataFrame area
  crs_map = map_edges.crs
  # Given the pulled CRS type, re-project the coordinates onto the same coordinate system of the map
  origin_projection = current_coord.to_crs(crs=crs_map)
  destination_projection = destination_coord.to_crs(crs=crs_map)
  # Create a GeoDataFrame to store the route of the shortest path
  output_path = gpd.GeoDataFrame()
  # Store all nodes from the graph area without the edge nodes
  map_nodes = ox.graph_to_gdfs(graph_proj, edges=False)
  # Extract coordinate info from origin node
  for oidx, orig in origin_projection.iterrows():
      # Locate the graph node closest to the user's location coordinates
      # nearest_nodes uses k-d trees to find nearest neighbor
      nearest_origin_node = ox.nearest_nodes(G=graph_proj, X=orig.geometry.x, Y=orig.geometry.y)
      # Extract coordinate info from destination node
      for tidx, target in destination_projection.iterrows():
          # Locate the graph node closest to the user's destination coordinates
          # nearest_nodes uses k-d trees to find nearest neighbor
          nearest_target_node = ox.nearest_nodes(graph_proj, X=target.geometry.x, Y=target.geometry.y)
          # Use dijkstra's algorithm to find the shortest path
          shortest_path = nx.dijkstra_path(graph_proj,
                                            source=nearest_origin_node,
                                            target=nearest_target_node, weight='length')
          # Isolate only the nodes of this shortest path
          shortest_path_nodes = map_nodes.loc[shortest_path]
          # Format results into a linestring to be used for plotting the shortest path for demo purposes
          # List the nodes into a linestring so that they can be written into a new GeoDataFrame
          #return len(list(shortest_path_nodes))
          nodes_list = LineString(list(shortest_path_nodes.geometry.values))
          # Place results into a GeoDataFrame
          output_path = output_path.append([[nodes_list]], ignore_index=True)


  # Now the output nodes need to be formatted so that they can used as an output to the navigation system
  new_coord = pd.DataFrame(shortest_path_nodes[['lon', 'lat']])
  new_coord = new_coord[['lat', 'lon']]
  new_index = list(range(0, len(new_coord)))
  # rename the node number index
  new_coord['num'] = new_index
  new_coord = new_coord.set_index('num')
  path_arr = new_coord.to_numpy()

  # This next section of code uses haversine formula to find the distance between two coordinates
  # First converts from degrees to radians
  # Iterates over the path array to add up all path lengths
  # Outputs path length in miles

  # Convert locations to radians
  current_lat = coord1 * (math.pi / 180)
  current_lon = coord2 * (math.pi / 180)
  destination_lat = coord3 * (math.pi / 180)
  destination_lon = coord4 * (math.pi / 180)
  # first node on the graph path
  lat1 = path_arr[0, 0] * (math.pi / 180)
  lon1 = path_arr[0, 1] * (math.pi / 180)
  # Find the length of the total path
  # Add together the distance from starting location to the first path node
  total_dist_mi = 3958.8 * 2 * asin(
    sqrt(sin((lat1 - current_lat) / 2) ** 2 + cos(current_lat) * cos(lat1) * sin((lon1 - current_lon) / 2) ** 2))
  # Compute distance between each node, then sum up the distances
  for i in range(1, (int(path_arr.size / 2))):
      # Update next destination lat and long points
      lat2 = path_arr[i, 0] * (math.pi / 180)
      lon2 = path_arr[i, 1] * (math.pi / 180)
      # Compute individual node distance
      dist = 3958.8 * 2 * asin(sqrt(sin((lat2 - lat1) / 2)
                                    ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2))
      total_dist_mi = total_dist_mi + dist
      # Update next location lat and long points
      lat1 = path_arr[i, 0] * (math.pi / 180)
      lon1 = path_arr[i, 1] * (math.pi / 180)
  # Add together the distance from destination location to the last path node
  total_dist_mi = total_dist_mi + (3958.8 * 2 * asin(sqrt(
    sin((destination_lat - lat1) / 2) ** 2 + cos(lat1) * cos(destination_lat) * sin(
      (destination_lon - lon1) / 2) ** 2)))

  # This next section of code writes the output path and length of the path to a csv and upload it to S3
  temp_csv_file = csv.writer(open("/tmp/path_coordinates.csv", "w+"))
  len_arr = np.array([[round(total_dist_mi,2),0]])
  path_arr = np.concatenate([len_arr, path_arr])
  np.savetxt('/tmp/path_coordinates.csv', path_arr, delimiter=",")
  np.savetxt('/tmp/path_coordinates.csv', path_arr, fmt="%f", delimiter=",")
  BUCKET_NAME = 'user-input-image'
  client = boto3.client('s3')
  client.upload_file('/tmp/path_coordinates.csv', BUCKET_NAME,'public/path_coordinates.csv')

  # Update a csv file in S3 to tell Android studio that the path_coordinates.csv is uploaded(a check buffer)
  s3 = boto3.resource('s3')
  s3.Bucket('user-input-image').download_file('public/comp-state.csv', '/tmp/comp-state.csv') 
  CSVData = open('/tmp/comp-state.csv')
  state_var = [1]
  np.savetxt('/tmp/comp-state.csv', state_var, delimiter=",")
  client = boto3.client('s3')
  client.upload_file('/tmp/comp-state.csv', BUCKET_NAME,'public/comp-state.csv')

  return path_arr.tolist(), total_dist_mi
