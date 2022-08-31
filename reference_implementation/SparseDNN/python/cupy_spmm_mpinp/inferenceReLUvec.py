import time
import numpy as np
import cupy as cp

# @profile
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = cp.array(Y0, dtype=cp.float32, order='f')


    # spmm time list
    spmmTimes = []

    # loop through each weight layer W[i]
    for i in range(len(W)):
        
        # tic = time.perf_counter()
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        # print(Y)
        # print(W[i])
        # print(Y.size)
        # print(Y.count_nonzero(), W[i].count_nonzero())
        # print(Y.shape, W[i].shape)
        # load weight matrix to GPU
        W_sel = cp.sparse.csr_matrix(W[i], dtype=cp.float32)
        # print(W_sel.shape, Y.shape)

        tic = time.perf_counter();
        Y = cp.cusparse.spmm(a=W_sel, b=Y)
        spmmTime = time.perf_counter() - tic;
        spmmTimes.append(spmmTime)
        # print(Y.shape)

        # Apply bias to non-zero entries.
        Y = cp.where(Y < -bias,  0, cp.where(Y > YMAX - bias, YMAX, Y + bias))
        
        # eliminate zero entries
        # Y.eliminate_zeros()
        Y = cp.array(Y, order='f')

    challengeRunTime = np.sum(spmmTimes)
    # print('[INFO] Run time (sec): %f' %(challengeRunTime));

    return Y, challengeRunTime