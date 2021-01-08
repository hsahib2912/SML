import struct as st
import numpy as np
import idx2numpy as hi
#import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy.linalg as npla

tr_img=hi.convert_from_file('train-images.idx3-ubyte')
tr_lab=hi.convert_from_file('train-labels.idx1-ubyte')
te_img=hi.convert_from_file('t10k-images.idx3-ubyte')
te_lab=hi.convert_from_file('t10k-labels.idx1-ubyte')

plt.imshow(tr_img[69],'gray')
plt.show()

def get_gauss(mean,var):
	gauss = np.full((28,28),0)
	for i in range(28):
		for j in range(28):
			gauss[i][j]=np.random.normal(mean,var)
	return gauss

def add_noise(mean,var,five_img):
	a=get_gauss(mean,var)
	for i in range(60000):
		five_img[i]=five_img[i]+a

	for i in range(5):
		show_img(five_img[i])

def mean(images):
	sum_img = np.full((28,28),0)
	for i in range(60000):
		sum_img+=images[i]
	return sum_img/60000

def compute_covar(five_img,mean):
	covar = np.full((28,28),0)
	for i in range(60000):
		five_img[i]= five_img[i]-mean
		covar+=np.dot(five_img[i],np.transpose(five_img[i]))

	covar=covar/59999
	return covar

def compute_eigen(covar):
	e_value,e_vector = npla.eig(covar)
	print(e_value.shape)
	print(e_vector.shape)
	return e_value,e_vector

def compute_X(e_value,e_vector):
	pass

def show_img(mat):
	plt.imshow(mat,'gray')
	plt.show()




five_img = np.full((60000,28,28),1)
for i in range(60000):
	five_img[i]=tr_img[i]
add_noise(0,50,five_img)
mean = mean(five_img)

show_img(mean)

covar = compute_covar(five_img,mean)

e_value,e_vector = compute_eigen(covar)

compute_X(e_value,e_vector)

