#Python 2.7.12

import numpy as np
import activation_functions
import error_functions

#################### map string name to function  ####################
#don't hesitate, follow the lines 27,28 and 79,80 for usage :)
#or you can ignore this part and handle function name by if ....

diff_activation_function_map={}
activation_function_map={}
loss_function_map={}

activation_function_map["softmax"] = activation_functions.softmax
activation_function_map["sigmoid"] = activation_functions.sigmoid
activation_function_map["identity"] = activation_functions.identity
activation_function_map["relu"] = activation_functions.relu

diff_activation_function_map["relu"] = activation_functions.drelu
diff_activation_function_map["sigmoid"] = activation_functions.dsigmoid
diff_activation_function_map["identity"] = activation_functions.didentity

loss_function_map["softmax_ce"] = error_functions.softmax_ce_derivation

#sample usage:
softmax = activation_function_map["softmax"]
#now you can use softmax(x) as regular function :)

########################################################################


def NN_output(config,w_list,b_list,x):
#output of given neural network whith input x
    last_out=x
    for i in xrange(len(config)):
        act=activation_function_map[config[i]["act_name"]]
        last_out=act(w_list[i]*last_out+b_list[i])
    return last_out

def NN_softmax_output_and_error(config,w_list,b_list,x):
    global softmax
    return softmax(NN_output(config,w_list,b_list,x))

def compute_gradient(config,w_list,b_list,loss_function_name,x,y):
    """
    * config is an input which determine each layer's characteristics
        config[i] shows the configuration of the ith layer
        config[i]["num"] determine number of neurons in the ith layer
        config[i]["act_name"] : activation function of ith layer
            pissible values for config[i]["act_name"] :
                "sigmoid"
                "relu"
                "identity"
        w_list[i] and b_list[i] is given to function as weight and baios parameters between i and i+1 layer
          notice that i starts from 0;

    * loss_function_name : name of loss_function as string
        possible values:
            "softmax_ce"

      * x : network input
      * y : target output

    you can see a sample usage in "train.py" file at line 17
    """
    z=[]  #value of layer befor activation function
    o=[x] #output of layer

    #forward:
    for i in xrange(len(config)):
        act=activation_function_map[config[i]["act_name"]] # you can use "act(input)"

        ### YOUR CODE HERE:
        z.append(w_list[i]*o[i]+b_list[i])
        o.append(act(z[i]))
        ### END YOUR CODE

    #backward:
    loss_function=loss_function_map[loss_function_name]
    dE_dlastO=loss_function(o[-1],y) #dE_dlastO is derivative of loss respect to output

    dE_dw=[]
    dE_db=[]
    dE_dO = np.transpose(dE_dlastO)
    
    for i in xrange(len(config)-1,-1,-1):
        d_act=diff_activation_function_map[config[i]["act_name"]]
        
        ### YOUR CODE HERE:
        dE_dz = np.array(dE_dO) * np.array(d_act(z[i]))
        dE_dw.append(np.matmul(dE_dz, o[i].T))
        dE_db.append(dE_dz)
        dE_dO = np.matmul(w_list[i].T, dE_dz)
        ### END YOUR CODE

    return o[-1],dE_dw,dE_db 
    # dE_dw have same structure as w_list
    # dE_db have same structure as b_list
    