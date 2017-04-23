__author__ = 'Donnie Kim and Chao Zhou'
import pandas as pd
import numpy as np
import scipy.io
import numpy as np
import scipy.misc as smp
from PIL import Image
import pickle

dictionary = {'label': int}
tag = 'pixel'
for i in range(3072):
    tag_num = tag + str(i)
    dictionary[tag_num] = int 

# Read X and y from the dataset (train and test)
print("reading extra")
# data_train = pickle.load(open("extra.p", "rb"))
data_train = pd.read_csv('extra.csv',dtype = dictionary)
print("finished reading")
# pickle.dump(data_train, open("extra.p", "wb"), protocol=4)

columns_train = data_train.columns
y_train = data_train.label.values
X_train = data_train[columns_train[1:]].values
X_train = X_train.reshape(32, 32, 3, -1).transpose(3,0,1,2)
# # To reverse the transpose, we simply need to run the following line
# X_train = X_train.transpose(1, 2, 3, 0)
# X_train = X_train.reshape((8, 3072))


num_train = X_train.shape[0]
dim_train = X_train.shape[1]

# Create a 32x32x3 array of 8 bit unsigned integers and save them all as 'image_index'_'label'.png
print("Initializing variables")
tempImg = np.zeros( ( 32,32,3), dtype=np.uint8 )
crop1 = np.zeros((num_train,28,28,3), dtype = np.uint8)
crop2 = np.zeros((num_train,28,28,3), dtype = np.uint8)
crop3 = np.zeros((num_train,28,28,3), dtype = np.uint8)
crop4 = np.zeros((num_train,28,28,3), dtype = np.uint8)
crop5 = np.zeros((num_train,28,28,3), dtype = np.uint8)
print("Looping")
for n in range(num_train):
	if (n % 100 == 0):
		print(str(n) + "/" + str(num_train))
	for i in range(dim_train):
		for j in range(dim_train):
			tempImg[i][j] = [ X_train[n][i][j][0], X_train[n][i][j][1], X_train[n][i][j][2] ]
	# 5 direction crop
	crop1[n] = tempImg[0:28, 0:28]
	crop2[n] = tempImg[0:28, 4:32]
	crop3[n] = tempImg[4:32, 0:28]
	crop4[n] = tempImg[4:32, 4:32]
	crop5[n] = tempImg[2:30, 2:30]

crop1 = crop1.transpose(1,2,3,0).reshape((num_train, 2352))
output_crop1 = np.zeros((crop1.shape[0], crop1.shape[1]+1), dtype = np.uint8)
for n in range(crop1.shape[0]):
	output_crop1[n][0] = y_train[n]
	output_crop1[n][1:] = crop1[n]
print("Outputting to cropped_extra_upperleft.csv")
df = pd.DataFrame(data=output_crop1, columns=columns_train[:2353])
df.to_csv("cropped_extra_upperleft.csv", index=None)
# Upper left


crop2 = crop2.transpose(1,2,3,0).reshape((num_train, 2352))
output_crop2 = np.zeros((crop2.shape[0], crop2.shape[1]+1), dtype = np.uint8)
for n in range(crop2.shape[0]):
	output_crop2[n][0] = y_train[n]
	output_crop2[n][1:] = crop2[n]
print("Outputting to cropped_extra_lowerleft.csv")
df = pd.DataFrame(data=output_crop2, columns=columns_train[:2353])
df.to_csv("cropped_extra_lowerleft.csv", index=None)
# Lower left


crop3 = crop3.transpose(1,2,3,0).reshape((num_train, 2352))
output_crop3 = np.zeros((crop3.shape[0], crop3.shape[1]+1), dtype = np.uint8)
for n in range(crop3.shape[0]):
	output_crop3[n][0] = y_train[n]
	output_crop3[n][1:] = crop3[n]
print("Outputting to cropped_extra_upperright.csv")
df = pd.DataFrame(data=output_crop3, columns=columns_train[:2353])
df.to_csv("cropped_extra_upperright.csv", index=None)
# Upper right


crop4 = crop4.transpose(1,2,3,0).reshape((num_train, 2352))
output_crop4 = np.zeros((crop4.shape[0], crop4.shape[1]+1), dtype = np.uint8)
for n in range(crop4.shape[0]):
	output_crop4[n][0] = y_train[n]
	output_crop4[n][1:] = crop4[n]
print("Outputting to cropped_extra_lowerright.csv")
df = pd.DataFrame(data=output_crop4, columns=columns_train[:2353])
df.to_csv("cropped_extra_lowerright.csv", index=None)
# Lower right


crop5 = crop5.transpose(1,2,3,0).reshape((num_train, 2352))
output_crop5 = np.zeros((crop5.shape[0], crop5.shape[1]+1), dtype = np.uint8)
for n in range(crop5.shape[0]):
	output_crop5[n][0] = y_train[n]
	output_crop5[n][1:] = crop5[n]
print("Outputting to cropped_extra_middle.csv")
df = pd.DataFrame(data=output_crop5, columns=columns_train[:2353])
df.to_csv("cropped_extra_middle.csv", index=None)
# Middle
