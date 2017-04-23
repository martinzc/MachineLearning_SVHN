__author__ = 'Donnie Kim and Chao Zhou'

import pandas as pd
import scipy.io
import numpy as np
import scipy.misc as smp
from PIL import Image


# Read X and y from the dataset

dictionary = {'label': int}
tag = 'pixel'
for i in range(3072):
    tag_num = tag + str(i)
    dictionary[tag_num] = int 

# Read X and y from the dataset (train and test)
print("reading extra")
data_train = pd.read_csv('cropped_extra_upperleft.csv',dtype = dictionary)

y_train = data_train.label.values
X_train = data_train[data_train.columns[1:]].values
print(X_train.shape)
X_train = X_train.reshape(28, 28, 3, -1).transpose(3,0,1,2)
# num_train = X_train.shape[0]
num_train = 1
dim_train = X_train.shape[1]

# Create a 32x32x3 array of 8 bit unsigned integers and save them all as 'image_index'_'label'.png
tempImg = np.zeros( (dim_train,dim_train,3), dtype=np.uint8 )

for n in range(num_train):
	for i in range(dim_train):
		for j in range(dim_train):
			tempImg[i][j] = [ X_train[n][i][j][0], X_train[n][i][j][1], X_train[n][i][j][2] ]
	img = Image.fromarray(tempImg, 'RGB')
	# img.save("svhn"+str(i)+".jpg")
	# img = smp.imsave('/Users/Donnie/PycharmProjects/Comp540/Project/data_images/'+str(n)+'_'+str(y_train[n])+'.png', tempImg )       
	# Create a PIL image
	img.show()                      # View in default viewer