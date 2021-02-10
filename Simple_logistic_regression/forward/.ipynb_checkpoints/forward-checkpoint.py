import numpy as np
from ..loss import binary_cross_entropy
from ..activation import sigmoid

def one_step_forward(w,b,X,Y):
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    a = sigmoid(z)
    loss = binary_cross_entropy(Y,a)
    cost = (1/m)*np.sum(loss)
    
    return z,a,cost
    