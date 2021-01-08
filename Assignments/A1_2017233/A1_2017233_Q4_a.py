#	HARKISHAN SINGH : 2017233

import random
import numpy as np
import matplotlib.pyplot as plt


print("\n\n\t\tQUESTION 4 : part a)")
d=2
covar=np.empty([d,d],dtype=float)
mean=[]

# creating a random covariance matrix (sigma matrix)
for i in range(d):
	for j in range(d):
		if i==j:
			covar[i][j]=random.randint(1,100)
		elif(i<j):
			covar[i][j]=random.randint(-100,100)
		else:
			covar[i][j]=covar[j][i]

print("\n\n\t\tRandomly generated covariance matrix is : \n",covar)
# creating a random mean vector
for i in range(d):
	mean.append(random.randint(-10,10))

print("\n\n\t\tRandomly generated mean vector is : \n",mean)
x,y = np.random.multivariate_normal(mean,covar,5000).T
plt.title("5000 points are plotted which are withdrawn from randomly generated covariance matrix and mean vector")
plt.scatter(x,y)
plt.show()
