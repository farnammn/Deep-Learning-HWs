#Python 2.7.12

import numpy as np
import random
import error_functions
import gradient


feture_number=20
class_number=3

def make_good_format(x,y):
	global feture_number
	label_one_hot=np.matrix(np.zeros((class_number,1))).astype("double")
	label_one_hot[y,0]=1.0
	inp_as_vector=np.reshape(x,(feture_number,1)).astype("double")
	return inp_as_vector,label_one_hot

def shuffle_pair(a,b):
	tmp = list(zip(a,b))
	random.shuffle(tmp)
	return zip(*tmp)

def load_dataset():
	
	dataset_x_tmp=np.load("synthetic_dataset_x.npy")	
	dataset_y_tmp=np.load("synthetic_dataset_y.npy")	

	dataset_x=[] #list of numpy matrix 20*1
	dataset_y=[] #list of numpy matrix 3*1  (one-hot encoded)
	for i in range(len(dataset_x_tmp)):
		x,y=make_good_format(dataset_x_tmp[i],dataset_y_tmp[i])
		dataset_x.append(x)
		dataset_y.append(y)


	dataset_x,dataset_y=shuffle_pair(dataset_x,dataset_y)

	dataset_x_tarin,dataset_y_train=dataset_x[6000:],dataset_y[6000:]
	dataset_x_test,dataset_y_test=dataset_x[:6000],dataset_y[:6000]

	return dataset_x_tarin,dataset_y_train,dataset_x_test,dataset_y_test

def accuracy(config,w_list,b_list,x_list,y_list):
	global class_number

	cc=0
	ca=0
	error_sum=0.0
	for i in range(len(x_list)):
		out=gradient.NN_softmax_output_and_error(config,w_list,b_list,x_list[i])
		error_sum+=error_functions.ce_error(out,y_list[i])
		m=0
		for j in range(1,class_number):
			if out[j]>out[m]:
				m=j
		if y_list[i][m,0]==1:
			cc+=1
		ca+=1
	ca=float(ca)
	cc=float(cc)
	return int((100.0*cc)/ca),error_sum/ca



