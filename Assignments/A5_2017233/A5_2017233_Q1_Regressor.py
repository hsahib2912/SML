import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor


def get_data(name):

	x = []
	y_label = []

	with open(name,'r') as file:
		readcsv = csv.reader(file)
		for row in readcsv:

			try:
				y_label.append(float(row[4]))
				x.append([float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[5]),float(row[6]),float(row[7]),float(row[9]),float(row[10]),float(row[11])])

			except Exception as e:
				pass



	return x,y_label

x_train,y_train_label = get_data("train_data.csv")
x_test,y_ground_truth = get_data("test_data.csv")

def compute_accuracy(y1,y2):
	mse = 0.

	for i in range(len(y1)):
		#print(y1[i]," ",y2[i])
		mse+= np.power(y1[i]-y2[i],2)

	mse = mse/len(y1)
	return mse

def DTR(depth):
	nodes = DecisionTreeRegressor(max_depth = depth)
	nodes = nodes.fit(x_train,y_train_label)
	y_prediction = nodes.predict(x_test)
	print(100*compute_accuracy(y_ground_truth,y_prediction))
	return 100*compute_accuracy(y_ground_truth,y_prediction)

def BDTR(sample):
	nodes = BaggingRegressor(max_samples = sample)
	nodes = nodes.fit(x_train,y_train_label)
	y_prediction = nodes.predict(x_test)
	return 100*compute_accuracy(y_ground_truth,y_prediction)

def random_forest(est):
	nodes = RandomForestRegressor(n_estimators = est)
	nodes = nodes.fit(x_train,y_train_label)
	y_prediction = nodes.predict(x_test)
	return (100*compute_accuracy(y_ground_truth,y_prediction))

def graph(x,y):
	plt.plot(x,y, marker='o',markerfacecolor='red')
	plt.title("Random Forest Regressor")
	plt.xlabel("Number of trees in forest")
	plt.ylabel("Mean Squared Error")
	plt.show()

x = [i for i in range(1,50)]
mse = []
for i in x:
	print(i)
	mse.append(random_forest(i))
graph(x,mse)




