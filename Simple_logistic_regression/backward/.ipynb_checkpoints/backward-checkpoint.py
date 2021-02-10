import numpy as np

def one_step_backward(a,Y,X):
    m = X.shape[1]
    dw= (1/m)*np.dot(a-Y,X.T)
    db= np.mean(a-Y)
    
    return dw, db