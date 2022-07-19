import numpy as np
import cupy as cp
import time

def inferenceReLUvec(W, Y0, spmm_time, batch_size=1):
    
    YMAX = 32 # set max value

    Y = Y0

    bias = -0.3
    if(Y.shape[0] == 4096):
        bias = -0.35
    elif(Y.shape[0] == 16384):
        bias = -0.4
    elif(Y.shape[0] == 65536):
        bias = -0.45

    tmp = int(Y0.shape[1]/batch_size)
    Yt = None

    for j in np.arange(0, Y0.shape[1], Y0.shape[1]/batch_size):
        j = int(j)
        Y = Y0[:,j:j + tmp]
        # loop through each weight layer W[i]
        Y = cp.array(Y, order='f')
        for i in range(len(W)):
            
            # % Propagate through layer.
            # % Note: using graph convention of A(i,j) means connection from i *to* j,
            # % that requires *left* multiplication feature *row* vectors.
            w = cp.sparse.csr_matrix(W[i])
            s = time.time()
            Y = cp.cusparse.spmm(w, Y)
            elapsed = time.time() - s
            spmm_time[0] += elapsed

            # Apply bias to non-zero entries.
            #Y = Z + cp.multiply(Z.astype('bool'), b)

            # Threshold negative values.
            Y = cp.where(Y < - bias,  0, cp.where(Y > YMAX - bias, YMAX, Y + bias))
            # Threshold maximum values.
            #Y[cp.where(Y > YMAX)] =  YMAX

            Y = cp.array(Y, order='f')
        if Yt is None:
            Yt = Y
        else:
            Yt = cp.concatenate((Yt, Y), axis=1)

    return Yt
