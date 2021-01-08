#	HARKISHAN SINGH : 2017233
import random
import numpy as np

print("QUESTION 4 : part b)")
d=int(input("\n\n\t\tEnter dimentions (single integer value) : \t"))
prob = float(input("\n\n\t\tEnter probability (from 0 to 1) : \t")) 
sigma = float(input("\n\n\t\tEnter sigma value : \t"))
print("\n\n\t\tEnter ",d," values of x")
x=[]
for i in range(d):
	x.append(float(input()))


# creating a random mean vector of dimention d 
mean=[]
for i in range(d):
	mean.append(random.randrange(-10,10))
	
# Computing discriminatnt function

dis = np.log(prob)
for i in range(d):
	x[i]-=mean[i]
dis+= -0.5*np.power(np.linalg.norm(x),2)/np.power(sigma,2)

print("\n\n\t\tValue of discriminant function is = ",dis)


