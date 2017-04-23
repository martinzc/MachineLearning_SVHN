__author__ = 'Donnie Kim and Chao Zhou'

import pandas as pd
import numpy as np
import numpy as np
from PIL import Image

# Read X and y from the dataset
data_train = pd.read_csv('test.csv')

columns_train = data_train.columns
X_train = data_train.values
X_train = X_train.reshape(32, 32, 3, -1).transpose(3,0,1,2)
print(X_train.shape)
# # To reverse the transpose, we simply need to run the following line
# X_train = X_train.transpose(1, 2, 3, 0)
# X_train = X_train.reshape((8, 3072))

num_train = X_train.shape[0]
# num_train = 5
dim_train = X_train.shape[1]

# Create a 32x32x3 array of 8 bit unsigned integers and save them all as 'image_index'_'label'.png
tempImg = np.zeros( ( 32,32,3), dtype=np.uint8 )
crop = np.zeros((num_train*5,28,28,3), dtype = np.uint8)
for n in range(num_train):
	if (n % 100 == 0):
		print(str(n) + "/" + str(num_train))
	for i in range(dim_train):
		for j in range(dim_train):
			tempImg[i][j] = [ X_train[n][i][j][0], X_train[n][i][j][1], X_train[n][i][j][2] ]

	# 5 direction crop
	crop[n*5+0] = tempImg[0:28, 0:28]
	crop[n*5+1] = tempImg[0:28, 4:32]
	crop[n*5+2] = tempImg[4:32, 0:28]
	crop[n*5+3] = tempImg[4:32, 4:32]
	crop[n*5+4] = tempImg[2:30, 2:30]

crop = crop.transpose(1,2,3,0).reshape((num_train*5, 2352))
print("Outputting to cropped_test.csv")

df = pd.DataFrame(data=crop, columns=columns_train[:2352])
df.to_csv("cropped_test.csv", index=None)

