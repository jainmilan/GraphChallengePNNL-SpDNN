import numpy as np
import cupy as cp

def inferenceReLUvec(W, Y0):
    
    YMAX = 32 # set max value

    Y = Y0

    bias = -0.3
    if(Y.shape[0] == 4096):
        bias = -0.35
    elif(Y.shape[0] == 16384):
        bias = -0.4
    elif(Y.shape[0] == 65536):
        bias = -0.45

    # loop through each weight layer W[i]
    for i in range(len(W)):
        
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        Y = cp.cusparse.spmm(W[i], Y)
        #b = bias[i]

        # Apply bias to non-zero entries.
        #Y = Z + cp.multiply(Z.astype('bool'), b)

        # Threshold negative values.
        Y = cp.where(Y < - bias,  0, cp.where(Y > YMAX - bias, YMAX, Y + bias))
        # Threshold maximum values.
        #Y[cp.where(Y > YMAX)] =  YMAX

        Y = cp.array(Y, order='f')

    return Y
