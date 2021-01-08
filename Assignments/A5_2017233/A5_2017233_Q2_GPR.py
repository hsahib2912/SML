import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt


x_data = np.linspace(0,11,12)
#print("x (Distance) = ",x_data)
x_data = x_data.reshape(-1,1)
y_data = np.array([-45,-51,-58,-63,-36,-52,-59,-62,-36,-43,-55,-64])
#print("y (Signal Strength) = ",y_data)

x_train = np.array([0,2,4,6,8,10,11])
#print("x (Distance) = ",x_train)
x_train = x_train.reshape(-1,1)
y_train = [-45,-58,-36,-59,-36,-55,-64]
#print("y (Signal Strength) = ",y_train)

x_test = np.array([1,3,5,7,9])
#print("x (Distance) = ",x_test)
x_test = x_test.reshape(-1,1)
y_truth = [-51,-63,-52,-62,-43]
#print("y (Signal Strength) = ",y_truth)

 
gp = GaussianProcessRegressor()

gp.fit(x_train,y_train)
y_pred,sigma = gp.predict(x_test,return_std = True)
print("Prediction = ",y_pred)
print("Variance = ",sigma)
plt.errorbar(x_test,y_pred,yerr = sigma,fmt='r.', markersize=10,label = "Prediction")
plt.plot(x_data,y_data,label = "Real Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
#plt.show()