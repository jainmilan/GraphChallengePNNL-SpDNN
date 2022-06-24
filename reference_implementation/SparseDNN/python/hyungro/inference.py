import numpy as np
import cupy as cp

def inferenceReLUvec(W, bias, Y0):
    
    YMAX = 32 # set max value

    Y = Y0

    # loop through each weight layer W[i]
    for i in range(len(W)):
        
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        dY = cp.array(Y, order='f')
        # (60000,1024),  (1024, 1024)
        Z = cp.cusparse.spmm(W[i], dY)#, transa=False, transb=False)
        b = bias[i]

        # Apply bias to non-zero entries.
        Y = Z + cp.multiply(Z.astype('bool').astype('float'), b)

        # Threshold negative values.
        cp.where(Y < 0, 0, Y)
        # Threshold maximum values.
        cp.where(Y > YMAX, YMAX, Y)

    return Y
