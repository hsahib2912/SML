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

def compute_mean(tr_img):
	tr_vector=np.full((60000,784),0)
	for i in range(60000):
		a=tr_img[i].flatten()
		tr_vector[i]=a

	sum_vector=np.full((1,784),0)
	for i in range(60000):
		sum_vector[0]=sum_vector[0]+tr_vector[i]

	tr_mean_vector = sum_vector[0]/60000
	return tr_mean_vector,tr_vector

def compute_dot(v1,v2):
	v=np.full((784,784),0)
	for i in range(784):
		for j in range(784):
			v[i][j]=v1[i]*v2[j]
	return v


def compute_covariance(tr_mean_vector,tr_vector):
	tr_covariance = np.full((784,784),0)
	for i in range(10):
		a=compute_dot(tr_vector[i]-tr_mean_vector,tr_vector[i]-tr_mean_vector)
		tr_covariance = tr_covariance + a

	plt.imshow(tr_covariance,'binary')
	plt.show()
	return tr_covariance/59999


def PCA(tr_covariance):
	eigvalues,eigvectors = npla.eig(tr_covariance)
	print(eigvalues.shape)
	print(eigvectors.shape)
	print("Eigenvalues = ",eigvalues)
	return eigvalues,eigvectors

def FDA(tr_mean_vector,tr_covariance):
	in_class_mean = np.full((784,784),0)
	print(tr_mean_vector.shape)
	in_class_mean = in_class_mean + np.dot(tr_mean_vector,np.transpose(tr_mean_vector))
	print(in_class_mean.shape)
	print(in_class_mean)
	print("Computing inverse")
	print(npla.inv(in_class_mean))
	mat = np.dot(in_class_mean,tr_covariance)
	eigvalues,eigvectors=npla.eig(mat)
	max=0
	for i in eigvalues:
		if(i>max):
			max=i
	return max

def find_large(pc_eigvalue,pc_eigvector):
	m=0
	ind=0
	for i in range(784):
		if(pc_eigvalue[i]>m):
			print("HI")
			ind=i
			m=pc_eigvalue[i]
			print("m = ",m)
			print("ind = ",ind)
	print(ind)
	v=pc_eigvector[ind]
	np.delete(pc_eigvalue,ind)
	np.delete(pc_eigvector,ind)
	return v





def visualize_PCA(pc_eigvalue,pc_eigvector):
	x=[]
	for i in range(784):
		x.append(i)
	v1=pc_eigvector[0]
	v2=pc_eigvector[1]
	hello=np.reshape(v1,(28,28))
	plt.imshow(hello,'gray')
	plt.title("Mean Image")
	plt.show()
	for i in range(784):
		plt.scatter(x,pc_eigvector[i])
	plt.show()

def eigen_energy(pc_eigvalue,tres):
	sum=0
	for i in pc_eigvalue:
		sum+=i

	st=0
	for i in range(784):
		if(st/sum>=tres):
			p=i
			break
		st+=pc_eigvalue[i]
	print(p)

tr_mean_vector,tr_vector = compute_mean(tr_img)
hello=np.reshape(tr_mean_vector,(28,28))
plt.imshow(hello,'gray')
plt.title("Mean Image")
plt.show()
tr_covariance = compute_covariance(tr_mean_vector,tr_vector)
pc_eigvalue,pc_eigvector = PCA(tr_covariance)
visualize_PCA(pc_eigvalue,pc_eigvector)
np.sort(pc_eigvalue)
print("Sorted  =. ",pc_eigvalue)
eigen_energy(pc_eigvalue,0.95)
eigen_energy(pc_eigvalue,0.70)
eigen_energy(pc_eigvalue,0.90)
eigen_energy(pc_eigvalue,0.99)
#FDA(tr_mean_vector,tr_covariance)
