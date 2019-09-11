#Python 2.7.12

import numpy as np
import etc
import gradient

feture_number=20
class_number=3

########################## load dataset ##########################

dataset_x_train,dataset_y_train,dataset_x_test,dataset_y_test=etc.load_dataset()
# dataset_x_tarin & dataset_x_test : list of numpy matrix 20x1
# dataset_y_tarin & dataset_y_test : list of numpy matrix 3x1  (one-hot encoded)

############################ making NN ############################
config=[]
w_list=[]
b_list=[]
#first layer
ln1=10
config.append({"num":ln1,"act_name":"sigmoid"})
w_list.append(np.matrix(np.random.normal(size=(ln1,feture_number),scale=0.2)).astype("double"))
b_list.append(np.matrix(np.zeros((ln1,1))).astype("double"))
#second layer
ln2=class_number
config.append({"num":ln2,"act_name":"identity"})
w_list.append(np.matrix(np.random.normal(size=(ln2,config[-2]["num"]),scale=0.2)).astype("double"))
b_list.append(np.matrix(np.zeros((ln2,1))).astype("double"))


accuracy_rate,ce_error=etc.accuracy(config,w_list,b_list,dataset_x_train,dataset_y_train)

############################ train NN ############################


### YOUR CODE HERE:
learning_rate = 0.1
data_size = len(dataset_x_train)
num_epochs = 100
batch_size = 50
num_batches = data_size // batch_size
dataset_x_train = np.array(dataset_x_train)
dataset_y_train = np.array(dataset_y_train)
for i in xrange(num_epochs):
    combined = zip(dataset_x_train, dataset_y_train)
    np.random.shuffle(combined)
    dataset_x_train, dataset_y_train = zip(*combined)
    
    epoch_corr = 0
    for j in xrange(num_batches):
        dw = [0 for s in range(len(config))]
        db = [0 for s in range(len(config))]
        for k in xrange(batch_size):
            index = j * batch_size + k
            y,dw_i,db_i=gradient.compute_gradient(config,w_list,b_list,"softmax_ce",dataset_x_train[index],dataset_y_train[index])
            dw += np.array(dw_i)
            db += np.array(db_i)
            y = np.argmax(y)
            y_true = np.argmax(dataset_y_train[index])
            epoch_corr += (y==y_true)  

        for k in xrange(len(config)-1,-1,-1):
            
            w_list[k] -= learning_rate * dw[len(config)-1 - k] / batch_size
            b_list[k] -= learning_rate * db[len(config)-1 - k] / batch_size
#         print w_list[0][0,0]
    print "epoch #{}, the accuraccy is {:.6f}".format(i, float(epoch_corr)/ data_size)


test_corr = 0
for i in xrange(len(dataset_x_test)):
    y =gradient.NN_output(config,w_list,b_list, dataset_x_test[i])
    y = np.argmax(y)
    y_true = np.argmax(dataset_y_test[i])
    test_corr += int(y==y_true)
    

print "test accuraccy is {:.6f}".format(float(test_corr)/ len(dataset_x_test))
    
# raise NotImplementedError
### END YOUR CODE
