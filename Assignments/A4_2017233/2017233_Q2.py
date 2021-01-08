import numpy as np 
import csv
import numpy.linalg as npla
import matplotlib.pyplot as plt

x=[]
y=[]
with open("data.csv",'r') as file:
	read = csv.reader(file,delimiter = ',')
	for lines in read:
		x.append(lines[0])
		y.append(lines[1])

def perform_k_fold(x,y):
	m = 1
	m_list = []
	error_list = []
	val_error = []
	while(m<20):
		x_train = np.ones((1,int(4./5.*len(x[0]))))
		y_train = np.ones((1,int(4./5.*len(y))))
		x_test = np.ones((1,int(1./5.*len(x[0]))))
		y_test = np.ones((1,int(1./5.*len(y))))
		
		#	FOLD on k = 5

		x_train[0] = x[0][:int(len(x[0])*4./5.)]
		x_test[0] =  x[0][int(len(x[0])*4./5.):]
		y_train[0] = y.T[0][:int(len(x[0])*4./5.)]
		y_test[0] =  y.T[0][int(len(x[0])*4./5.):]
		
		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e5 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali5 = get_error(w,x_test_mat,y_test.T)

		#	FOLD on k=4

		x_train[0] = np.concatenate((x[0][:int(len(x[0])*3./5.)],x[0][int(len(x[0])*4./5.):]))
		x_test[0] =  x[0][int(len(x[0])*3./5.):int(len(x[0])*4./5.)]
		y_train[0] = np.concatenate((y.T[0][:int(len(x[0])*3./5.)],y.T[0][int(len(x[0])*4./5.):]))
		y_test[0] =  y.T[0][int(len(x[0])*3./5.):int(len(x[0])*4./5.)]

		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e4 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali4 = get_error(w,x_test_mat,y_test.T)


		#	FOLD on k=3

		x_train[0] = np.concatenate((x[0][:int(len(x[0])*2./5.)],x[0][int(len(x[0])*3./5.):]))
		x_test[0] =  x[0][int(len(x[0])*2./5.):int(len(x[0])*3./5.)]
		y_train[0] = np.concatenate((y.T[0][:int(len(x[0])*2./5.)],y.T[0][int(len(x[0])*3./5.):]))
		y_test[0] =  y.T[0][int(len(x[0])*2./5.):int(len(x[0])*3./5.)]

		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e3 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali3 = get_error(w,x_test_mat,y_test.T)


		#	FOLD on k=2

		x_train[0] = np.concatenate((x[0][:int(len(x[0])*1./5.)],x[0][int(len(x[0])*2./5.):]))
		x_test[0] =  x[0][int(len(x[0])*1./5.):int(len(x[0])*2./5.)]
		y_train[0] = np.concatenate((y.T[0][:int(len(x[0])*1./5.)],y.T[0][int(len(x[0])*2./5.):]))
		y_test[0] =  y.T[0][int(len(x[0])*1./5.):int(len(x[0])*2./5.)]

		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e2 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali2 = get_error(w,x_test_mat,y_test.T)

		#	FOLD on k=1

		x_train[0] = x[0][int(len(x[0])*1./5.):]
		x_test[0] =  x[0][:int(len(x[0])*1./5.)]
		y_train[0] = y.T[0][int(len(x[0])*1./5.):]
		y_test[0] =  y.T[0][:int(len(x[0])*1./5.)]
		
		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e1 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali1 = get_error(w,x_test_mat,y_test.T)

		e_avg = (e1+e2+e3+e4+e5)/5
		vali_avg = (vali1+vali2+vali3+vali4+vali5)/5
 
		m_list.append(m)
		error_list.append(e_avg)
		val_error.append(vali_avg)

		m+=1

	plt.scatter(m_list,error_list)
	plt.scatter(m_list,val_error)
	plt.xlabel("Degree of polynomial/regression")
	plt.ylabel("RMSE")
	plt.legend(("Train Error","Validation error"))
	plt.show()

def create_x_mat(x,m):
	x_mat = np.ones((1+m,len(x[0])))
	for i in range(m):
		for j in range(len(x[0])):
			x_mat[i][j] = x[0][j]**(m-i) 

	return x_mat

def get_w(x,y):

	w = np.dot(x,x.T)
	try:
		w = npla.inv(w)
	except : 
		dia = np.identity(2)
		dia[0][0] = dia[1][1] = 0.01 
		w = np.add(w,dia)
		w = npla.inv(w)
	w = np.dot(w,x)
	w = np.dot(w,y)
	return w

def get_error(w,x_mat,y):

	t=0
	for i in range(len(x_mat[0])):
		y_pred = np.dot(x_mat[:,i],w)
		y_rel = y[i]
		t+=(y_pred-y_rel)**2
	
	t=t/len(x_mat[0])
	t=np.sqrt(t)
	return t


x.pop(0)
y.pop(0)

for i in range(len(x)):
	x[i] = float(x[i])
	y[i] = float(y[i])



train_size = int(0.8*len(x))
test_size = int(0.2*len(x))


x_train = np.array(x[:train_size])
x_train = np.reshape(x_train,(1,train_size))
x_test = np.array(x[:test_size])
x_test = np.reshape(x_test,(1,test_size))

y_train = np.array(y[:train_size])
y_train = np.reshape(y_train,(1,train_size))
y_test = np.array(y[:test_size])
y_test = np.reshape(y_test,(1,test_size))

perform_k_fold(x_train,y_train.T)





