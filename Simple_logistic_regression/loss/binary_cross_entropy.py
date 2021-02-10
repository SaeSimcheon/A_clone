import numpy as np

def binary_cross_entropy(y,a):
    return - np.multiply(y,np.log(a)) - np.multiply(1-y,np.log(1-a))