# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:28:35 2020

@author: sveta
"""

from minisom import MiniSom  

import matplotlib.pyplot as plt

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
for filename in glob.glob('horse2zebra/*.png'): 
    im=Image.open(filename)
    im.load()
    data = np.asarray(im, dtype= "int32")
    if "real" in filename:
       real_list.append(data.flatten())
       legend_real.append(filename)
    if "fake" in filename:
      fake_list.append(data.flatten())
      legend_fake.append(filename)
         

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

#set on which data to train on here
data = all_list

#parameter for som training is number of iterations, should be comparable to number of samples for best results
iteration = len(data)
  
#train som
som = MiniSom(10, 10, pixels, sigma=0.5, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, iteration)

#plt.pcolor(som.distance_map().T, cmap='bone_r')

#show some results

plt.figure(figsize=(8, 7))
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()

act = som.activation_response(data)
print(act)

print(som.quantization_error(data))

#print(som.win_map(data))