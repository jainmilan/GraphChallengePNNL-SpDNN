import sys
import time
import cupy as cp

# @profile
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = Y0

    # loop through each weight layer W[i]
    # print(Y.size)
    iteration_times = []
    # print(Y.shape, W[0].shape)
    # print(Y, W[0])
    # print(Y.size, W[0].size)
    # Z = Y.todense(order='f')
    first_iteration = True
    for i in range(len(W)):
        
        # tic = time.perf_counter()
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        
        # print(i)
        Z = cp.multiply(Y.todense(order='f'), bias[i])
        Z = cp.array(Z, order='f')
        # print(Z.size)
        Y = cp.sparse.csr_matrix(cp.cusparse.spmm(a=Y, b=W[i], c=Z, alpha=1, beta=1))
        # b_multiplier = Z
        # b_multiplier.data[cp.where(b_multiplier.data !=0 )] = 1
        # b_multiplier.eliminate_zeros()
        # b_multiplier
        # b = bias[i]
        
        # Apply bias to non-zero entries.
        # !!!! COULD BE MADE EFFICIENT
        # Y = Z + (Z.astype('bool').astype('float').multiply(b))
        # Y = Z + (b_multiplier.multiply(b))
        
        # print(cp.where(Y.data < 0))
        # Threshold negative values.
        Y = cp.where(Y < 0, 0, Y)
        # Y.data = cp.where(Y.data < 0, 0, Y.data)
        
        # Threshold maximum values.
        # Z[cp.where(Z > YMAX)] = YMAX
        Y.data = cp.where(Y.data > YMAX, YMAX, Y.data)
        
        Y.eliminate_zeros()
        # iteration_times.append(time.perf_counter()-tic)

    # print(cp.mean(cp.array(iteration_times)))
    return Y