from ..Utils import initializer
from ..loss import binary_cross_entropy
from ..forward import one_step_forward
from ..backward import one_step_backward


def optimize(X,Y,iteration,learning_rate):
    w, b=initializer(X.shape[0])
    costs = dict()
    for i in range(0,iteration): 
        z,a, cost = one_step_forward(w, b, X, Y)
        dw,db = one_step_backward(a,Y,X)
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs[i] = cost 
        print(str(i))
        print(cost)
    return w, b, costs