# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:28:35 2020

@author: sveta
"""

from minisom import MiniSom  

import matplotlib.pyplot as plt

import statistics

import os
import glob
from PIL import Image
import numpy as np


#for CIFAR 10 dataset
#file = "data_batch_1"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 
#labels = unpickle(file)
#print(labels.keys())
#data = labels[b'data']
#print(data)


#simple example
#data = [[ 1.00,  1.00,  1.00,  1.00],
 #       [ 0.82,  0.50,  0.23,  0.03],
  #      [ 0.80,  0.54,  0.22,  0.03],
   #     [ 0.80,  0.53,  0.26,  0.03],
    #    [ 0.00,  0.00,  0.00,  0.00],
     #   [ 0.75,  0.60,  0.25,  0.03],
      #  [ 0.77,  0.59,  0.22,  0.03]] 
   
real_list = []
fake_list = []
#store name files 
legend_fake = []
legend_real = []

dataset = "horse2zebra/*.png"

def read_data():
   print("results for dataset " + dataset)
   for filename in glob.glob(dataset): 
       im=Image.open(filename)
       im.load()
       data = np.asarray(im, dtype= "int32")
       filename = filename[22:-4]
       if "real" in filename:
          real_list.append(data.flatten())
          legend_real.append(filename)
       if "fake" in filename:
         fake_list.append(data.flatten())
         legend_fake.append(filename)
            
def real_data(): 
   data = real
   
   #parameter for som training is number of iterations, should be comparable to number of samples for best results
   iteration = len(data)
   #train som
   som = MiniSom(5, 5, pixels, sigma=0.5, learning_rate=0.5)
   som.random_weights_init(data)
   som.train_random(data, iteration)
   
   #plt.pcolor(som.distance_map().T, cmap='bone_r')
   
   #show some results
   
   print("results for real data\n")
   
   plt.figure(figsize=(8, 7))
   frequencies = som.activation_response(data)
   plt.pcolor(frequencies.T, cmap='Blues') 
   plt.colorbar()
   plt.show()
   
   act = som.activation_response(data)
   #print(act)
   
   #print(som.quantization_error(data))
   
   #print(som.topographic_error(data))
   
   #print(som.winner(data[1]))
   #som.labels_map(data, legend_real)
   results = []
   size = len(data)
   
   for i in range(0,size):
      results.append(som.winner(data[i]))

   pairs = []
   for i in range(0, size):
      pairs.append([])
      for j in range(0, size):
         if results[i]==results[j]:
            pairs[i].append(j)

   return pairs

#figure out runtime error for quan error
#better code structure for more tests
#saving results
#same initial weights??
   
def fake_data():
   data = fake
   
   #parameter for som training is number of iterations, should be comparable to number of samples for best results
   iteration = len(data)
   
   
   #train som
   som = MiniSom(5, 5, pixels, sigma=0.5, learning_rate=0.5)
   som.random_weights_init(data)
   som.train_random(data, iteration)
   
   #plt.pcolor(som.distance_map().T, cmap='bone_r')
   
   #show some results
   
   print("results for fake data\n")
   
   plt.figure(figsize=(8, 7))
   frequencies = som.activation_response(data)
   plt.pcolor(frequencies.T, cmap='Blues') 
   plt.colorbar()
   plt.show()
   
   act = som.activation_response(data)
   #print(act)
   
   #print(som.quantization_error(data))
   
   #print(som.topographic_error(data))
   
   #print(som.winner(data[1]))
   #som.labels_map(data, legend_fake)
   results = []
   size = len(data)
   
   for i in range(0,size):
      results.append(som.winner(data[i]))

   pairs = []
   for i in range(0, size):
      pairs.append([])
      for j in range(0, size):
         if results[i]==results[j]:
            pairs[i].append(j)

   return pairs

def combined_data():
   data = all_list
   
   #parameter for som training is number of iterations, should be comparable to number of samples for best results
   iteration = len(data)
   #train som
   som = MiniSom(5, 5, pixels, sigma=0.5, learning_rate=0.5)
   som.random_weights_init(data)
   som.train_random(data, iteration)
   
   #plt.pcolor(som.distance_map().T, cmap='bone_r')
   
   #show some results
   
   print("results for combined data\n")
   
   plt.figure(figsize=(8, 7))
   frequencies = som.activation_response(data)
   plt.pcolor(frequencies.T, cmap='Blues') 
   plt.colorbar()
   plt.show()
   
   act = som.activation_response(data)
   #print(act)
   
   #print(som.quantization_error(data))
   
   #print(som.topographic_error(data))
   
   #print(som.winner(data[1]))
   
   results = []
   
   for i in range(0,iteration):
      results.append(som.winner(data[i]))

   pairs = []
   for i in range(0, iteration):
      pairs.append([])
      for j in range(0, iteration):
         if results[i]==results[j]:
            pairs[i].append(j)

   return pairs

def compare(real, fake):
   print("here")
   res = []
   for i in range(0, len(real)):
      s1 = set(real[i])
      s2 = set(fake[i])
      res.append(len(s1.intersection(s2))/len(s1.union(s2)))
      
   print("mean Jaccard similiarity: " + str(statistics.mean(res)))
   print("median Jaccard similiarity: " + str(statistics.median(res)))
   #print("mode Jaccard similiarity: " + str(statistics.mode(res)))
   
   return res

   
read_data() 
   
#domain B
fake = np.array(fake_list)
#print(legend_fake)
#print(len(legend_fake))
#print(fake.shape)
   
#domain A
real = np.array(real_list)
#print(legend_real)
#print(len(legend_real))
#print(real.shape)
   
#combined dataset
images = []
images.extend(fake_list)
images.extend(real_list)
all_list = np.array(images)
   
legend = []
legend.extend(legend_fake)
legend.extend(legend_real)
#print(legend)
#print(len(legend))
#print(all_list.shape)

#number of parameters in a sample
pixels = len(fake[1])
#print(pixels)

iteration = 100

groups_real = real_data()
groups_fake = fake_data()

print(compare(groups_real, groups_fake))

res = combined_data()

size = len(res)

sum = 0
for i in range(0, int(size/2)):
   if int(i+50) in res[i]:
      sum = sum + 1
      
print(str(sum) +  " pairs were grouped together out of " + str(int(size/2)))
         
         



   