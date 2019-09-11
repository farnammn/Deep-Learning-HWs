#Python 2.7.12

import numpy as np
import activation_functions

def softmax_ce_derivation(y_hat,y):
#derivation of CE(softmax(y),y_hat) respect to y
	return (activation_functions.softmax(y_hat)-y).T


def ce_error(y_hat_normalized,y):
#Cross Entropy error :CE(y,y_hat)
	return 	-1.0*np.multiply(y,np.log(y_hat_normalized+1.0e-20)).sum()

##################### test case #####################
def test_softmax_ce_derivation():
	"""
	Some simple tests to get you started.
	Warning: these are not exhaustive.
	"""
	print "Running basic tests on softmax_ce_derivation..."
	for inp_y_hat,inp_y,out in ( 
		([0,0,1],[0,0,1],[0.21194156,0.21194156,-0.42388312]),
		([1,2,1],[0.21194155761708547,0.57611688476582912,0.21194155761708547],[0,0,0]),
		):
		inp_y_hat_matrix=np.matrix(inp_y_hat).reshape((3,1))
		inp_y_matrix=np.matrix(inp_y).reshape((3,1))
		test_matrix=np.matrix(out).reshape((1,3))
		ans=softmax_ce_derivation(inp_y_hat_matrix,inp_y_matrix)
		assert np.allclose(test_matrix, ans, rtol=1e-05, atol=1e-06)

if __name__ == "__main__":
	test_softmax_ce_derivation()
