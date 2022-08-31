import time
import numpy as np
import cupy as cp

# @profile
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = Y0

    # spgemm time list
    spgemmTimes = []
    
    # loop through each weight layer W[i]
    for i in range(len(W)):
        
        # tic = time.perf_counter()
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        tic = time.perf_counter();
        Y = Y @ W[i]
        spgemmTime = time.perf_counter() - tic;
        spgemmTimes.append(spgemmTime)

        # Apply bias to non-zero entries.
        Y.data = cp.where(Y.data < -bias,  0, cp.where(Y.data > YMAX - bias, YMAX, Y.data + bias))
        
        # eliminate zero entries
        Y.eliminate_zeros()

    challengeRunTime = np.sum(spgemmTimes)
    
    return Y, challengeRunTime