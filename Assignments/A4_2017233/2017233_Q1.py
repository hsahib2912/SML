import numpy as np 
import pandas
import numpy.linalg as npla
import matplotlib.pyplot as plt

def process_data():
	file = open("data.txt",'r')
	lines = file.readlines()
	data = []
	for i in range(len(lines)):
		lines[i] = lines[i].split(' ')
		c = lines[i].count('')
		for j in range(c):
			lines[i].remove('')

		lines[i][13] = lines[i][13].replace('\n','')

		for j in range(13):
			lines[i][j] = float(lines[i][j])

		data.append(lines[i])

	print("Processing done!!!")
	return data

def divide(data):
	k = int(0.8*len(data))
	train = data[:k]
	test = data[k:] 
	return train,test

def get_y(data):

	y = np.ones((len(data),1))
	for i in range(len(data)):
		y[i] = data[i][13]

	return y

def get_rmse(data,y):
	x_rmse = np.zeros((13,1))


	for i in range(13):
		x = np.ones((2,len(data)))
		for j in range(len(data)):
			x[0][j] = data[j][i]

		w = get_w(x,y)
		x_rmse[i] = compute_rmse_val(data,w,i)

	return x_rmse
		

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

def compute_rmse_val(data,w,i):

	w1 = w[0]
	w0 = w[1]
	t=0
	for j in range(len(data)):

		y_pred = w1*data[j][i]+w0
		y_rel = data[j][13]

		y_pred = float(y_pred)
		y_rel = float(y_rel)	
		a = (y_pred-y_rel)**2
		t+= a

	t = t/len(data)

	return np.sqrt(t)

def get_lstat(data):

	x = np.ones((1,len(data)))
	for i in range(len(data)):
		x[0][i] = data[i][12]
	
	return x


data = process_data()
train , test = divide(data)

train_y = get_y(train)
test_y = get_y(test)

train_rmse = get_rmse(train,train_y)
print("RMSE of training data = ",train_rmse)
test_rmse = get_rmse(test,test_y)
print("RMSE of testing data (20%) = ",test_rmse)

#			LSTAT

def perform_k_fold(x,y):
	m = 1
	m_list = []
	error_list = []
	val_error = []
	while(m<10):
		x_train = np.ones((1,int(4./5.*len(x[0]))))
		y_train = np.ones((1,int(4./5.*len(y))))
		x_test = np.ones((1,int(1./5.*len(x[0]))))
		y_test = np.ones((1,int(1./5.*len(y))))
		
		#	FOLD on k = 5

		x_train[0] = x[0][:int(len(x[0])*4./5.)]
		x_test[0] =  x[0][int(len(x[0])*4./5.)+1:]
		y_train[0] = y.T[0][:int(len(x[0])*4./5.)]
		y_test[0] =  y.T[0][int(len(x[0])*4./5.)+1:]
		
		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e5 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali5 = get_error(w,x_test_mat,y_test.T)

		#	FOLD on k=4

		x_train[0] = np.concatenate((x[0][:int(len(x[0])*3./5.)],x[0][int(len(x[0])*4./5.):]))
		x_test[0] =  x[0][int(len(x[0])*3./5.)+1:int(len(x[0])*4./5.)]
		y_train[0] = np.concatenate((y.T[0][:int(len(x[0])*3./5.)],y.T[0][int(len(x[0])*4./5.):]))
		y_test[0] =  y.T[0][int(len(x[0])*3./5.)+1:int(len(x[0])*4./5.)]

		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e4 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali4 = get_error(w,x_test_mat,y_test.T)


		#	FOLD on k=3

		x_train[0] = np.concatenate((x[0][:int(len(x[0])*2./5.)],x[0][int(len(x[0])*3./5.):]))
		x_test[0] =  x[0][int(len(x[0])*2./5.)+1:int(len(x[0])*3./5.)]
		y_train[0] = np.concatenate((y.T[0][:int(len(x[0])*2./5.)],y.T[0][int(len(x[0])*3./5.):]))
		y_test[0] =  y.T[0][int(len(x[0])*2./5.)+1:int(len(x[0])*3./5.)]

		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e3 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali3 = get_error(w,x_test_mat,y_test.T)


		#	FOLD on k=2

		x_train[0] = np.concatenate((x[0][:int(len(x[0])*1./5.)],x[0][int(len(x[0])*2./5.):]))
		x_test[0] =  x[0][int(len(x[0])*1./5.)+1:int(len(x[0])*2./5.)]
		y_train[0] = np.concatenate((y.T[0][:int(len(x[0])*1./5.)],y.T[0][int(len(x[0])*2./5.):]))
		y_test[0] =  y.T[0][int(len(x[0])*1./5.)+1:int(len(x[0])*2./5.)]

		x_mat = create_x_mat(x_train,m)
		w = get_w(x_mat,y_train.T)	
		e2 = get_error(w,x_mat,y_train.T)

		x_test_mat = create_x_mat(x_test,m)
		vali2 = get_error(w,x_test_mat,y_test.T)

		#	FOLD on k=1

		x_train[0] = x[0][int(len(x[0])*1./5.)+1:]
		x_test[0] =  x[0][:int(len(x[0])*1./5.)]
		y_train[0] = y.T[0][int(len(x[0])*1./5.)+1:]
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


def get_error(w,x_mat,y):

	t=0
	for i in range(len(x_mat[0])):
		y_pred = np.dot(x_mat[:,i],w)
		y_rel = y[i]
		t+=(y_pred-y_rel)**2
	
	t=t/len(x_mat[0])
	t=np.sqrt(t)
	return t


def create_x_mat(x,m):
	x_mat = np.ones((1+m,len(x[0])))
	for i in range(m):
		for j in range(len(x[0])):
			x_mat[i][j] = x[0][j]**(m-i) 

	return x_mat






x_train = get_lstat(train)
y_train = train_y

x_test = get_lstat(test)
y_test = test_y

print(y_train.shape)
perform_k_fold(x_train,y_train)


x_mat = create_x_mat(x_train,7)
print(x_mat.shape)
w = get_w(x_mat,y_train)
error = get_error(w,x_mat,y_train)
print("RMSE of whole training data = ",error)

x_mat = create_x_mat(x_test,7)
w = get_w(x_mat,y_test)
error = get_error(w,x_mat,y_test)
print("RMSE of whole testing data = ",error)



