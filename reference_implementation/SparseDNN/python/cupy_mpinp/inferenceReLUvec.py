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
        # print(Y)
        # print(W[i])
        # print(Y.size)
        # print(Y.count_nonzero(), W[i].count_nonzero())
        # print(Y.shape, W[i].shape)
        # load weight matrix to GPU
        W_sel = cp.sparse.csr_matrix(W[i], dtype=cp.float32)

        tic = time.perf_counter();
        Y = cp.cusparse.csrgemm2(Y, W_sel)
        spgemmTime = time.perf_counter() - tic;
        spgemmTimes.append(spgemmTime)

        # Apply bias to non-zero entries.
        Y.data = cp.where(Y.data < -bias,  0, cp.where(Y.data > YMAX - bias, YMAX, Y.data + bias))
        
        # eliminate zero entries
        Y.eliminate_zeros()

    challengeRunTime = np.sum(spgemmTimes)
    # print('[INFO] Run time (sec): %f' %(challengeRunTime));

    return Y, challengeRunTime