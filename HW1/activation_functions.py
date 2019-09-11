#Python 2.7.12

import numpy as np

def softmax(x): 
#softmax function at x
#input and output is a numpy matrix with size of [nx1]
    ### YOUR CODE HERE:
    softmax_offset = np.max(x)
    o = np.exp(x - softmax_offset) / np.sum(np.exp(x - softmax_offset))
    return o
    ### END YOUR CODE

def sigmoid(x):
#sigmoid function at x (element wise)
#input and output is a numpy matrix with size of [nx1]
    ### YOUR CODE HERE:
    o = np.zeros((x.size,1), float)
    for i, xi in enumerate(x):
        if xi > 0: o[i] = 1 / (1 + np.exp(-xi))
        else: o[i] = np.exp(xi) / (np.exp(xi) + 1)  
    return o
    ### END YOUR CODE


def dsigmoid(x):
#derivation of sigmoid function at x
    tmp=sigmoid(x)
    return tmp-np.power(tmp,2.0)


def relu(x):
#Rectified Linear Unit at x
	return (x+abs(x))/2.0
def drelu(x):
#derivation of Rectified Linear Unit at x
	return np.matrix(x>0)*1.0

def identity(x):
	return x
def didentity(x):
	return 1.0


##################### test case #####################
def test_softmax():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests on softmax...")
    for inp,out in (
        ([1,2],[0.26894142,0.73105858]),
        ([1001,1002],[0.26894142,0.73105858]),
        ([-1001,-1002],[0.73105858, 0.26894142])
        ):
        inp_matrix=np.matrix(inp).reshape((2,1))
        test_matrix=np.matrix(out).reshape((2,1))
        ans=softmax(inp_matrix)
        print(ans)
        assert np.allclose(test_matrix, ans, rtol=1e-05, atol=1e-06)
        
def test_sigmoid():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests on sigmoid...")
    for inp,out in ( 
        ([1,2],[0.73105858, 0.88079708]),
        ([-1,-2],[0.26894142, 0.11920292]),
        ([1001,-1001],[1.0,0.0]),
        ):
        inp_matrix=np.matrix(inp).reshape((2,1))
        test_matrix=np.matrix(out).reshape((2,1))
        ans=sigmoid(inp_matrix)
        assert np.allclose(test_matrix, ans, rtol=1e-05, atol=1e-06)

if __name__ == "__main__":
    test_softmax()
    test_sigmoid()