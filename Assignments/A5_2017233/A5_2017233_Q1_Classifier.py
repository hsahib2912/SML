import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import matplotlib.pyplot as plt

def get_data(name):

	x = []
	y_label = []

	with open(name,'r') as file:
		readcsv = csv.reader(file)
		for row in readcsv:

			try:
				x.append([float(row[0]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[9]),float(row[10]),float(row[11])])
				y_label.append(float(row[1]))

			except Exception as e:
				pass



	return x,y_label

def compute_accuracy(y1,y2):
	same = 0.

	for i in range(len(y1)):
		if(y1[i]==y2[i]):
			same+=1

	return(same/len(y1))

x_train,y_train_label = get_data("train_data.csv")
x_test,y_ground_truth = get_data("test_data.csv")




def DTC(depth):
	nodes = tree.DecisionTreeClassifier(max_depth = depth)
	nodes = nodes.fit(x_train,y_train_label)
	y_prediction = nodes.predict(x_test)

	return 100*compute_accuracy(y_ground_truth,y_prediction)

def BDTC(sample):
	nodes = BaggingClassifier(max_samples = sample)
	nodes = nodes.fit(x_train,y_train_label)
	y_prediction = nodes.predict(x_test)

	return 100*compute_accuracy(y_ground_truth,y_prediction)

def random_forest(est):
	nodes = RandomForestClassifier(n_estimators = est)
	nodes = nodes.fit(x_train,y_train_label)
	y_prediction = nodes.predict(x_test)
	return (100*compute_accuracy(y_ground_truth,y_prediction))

def graph(x,y):
	plt.plot(x,y, marker='o',markerfacecolor='red')
	plt.title("Random Forest Classifier")
	plt.xlabel("Number of trees in forest")
	plt.ylabel("Accuracy (in %)")
	plt.show()


x = [i for i in range(1,50)]
accuracy = []
for i in x:
	accuracy.append(random_forest(i))
graph(x,accuracy)





