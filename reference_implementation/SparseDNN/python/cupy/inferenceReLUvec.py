from dataclasses import replace
import sys
import time
import numpy as np
import cupy as cp
# from numba import jit
import numba as nb

# sparse_arr, YMAX
# @nb.njit(target_backend='cuda')
def replace_vals(sparse_arr, YMAX):
    # sys.exit(sparse_arr)
    # indx = 0
    print(sparse_arr.shape, sparse_arr.size)
    for i in nb.prange(sparse_arr.size):
        # sparse_arr[i] = max(sparse_arr[i], 0)
        # sparse_arr[i] = min(sparse_arr[i], YMAX)
        if sparse_arr[i] < 0:
            sparse_arr[i] = 0
        elif sparse_arr[i] > YMAX:
            sparse_arr[i] = YMAX
        
    return sparse_arr

update_kern = cp.RawKernel(r'''
extern "C" __global__
void update_vals(float* x1, float YMAX)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (x1[tid] < 0) {
        x1[tid] = 0;    
    } else if (x1[tid] > YMAX) {
        x1[tid] = YMAX;
    }
}
''', 'update_vals')




# module = cp.RawModule(code=loaded_from_source)
# ker_sum = module.get_function('test_sum')
# ker_times = module.get_function('test_multiply')

@cp.fuse()
def mult_add(Y, b):
    # Y_multiplier = Y
    # Y_multiplier.data = cp.where(Y_multiplier.data != 0, 1, Y_multiplier.data)
    # Y[Y.non]
    # b.data = cp.repeat(b.data, Y.shape[0], axis=0)
    # print(b.shape)
    # sys.exit(b.shape)
    # Y.data = cp.where(Y.data!=0, 1, Y.data)
    # return Y.multiply(b)
    # Y[Y.nonzero()] = 1
    # return Y.multiply(b)
    return Y.astype('bool').astype('float').multiply(b)

# @profile
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = Y0

    # loop through each weight layer W[i]
    # print(Y.size)
    # iteration_times = []
    # print(Y.shape, W[0].shape)
    # print(Y, W[0])
    # print(Y.size, W[0].size)
    # Z = Y
    for i in range(len(W)):
        
        # tic = time.perf_counter()
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        
        # print(i)
        Y = Y @ W[i]
        # b_multiplier = Z
        # b_multiplier.data[cp.where(b_multiplier.data !=0 )] = 1
        # b_multiplier.eliminate_zeros()
        # b_multiplier
        # b = bias[i]
        
        # Apply bias to non-zero entries.
        # !!!! COULD BE MADE EFFICIENT
        # Y = Z + (Z.astype('bool').astype('float').multiply(b))
        # Y = Z + (b_multiplier.multiply(b))
        # Z = Y.astype('bool').astype('float').multiply(bias[i])
        
        # Y = cp.cusparse.csrgemm2(Y, W[i])
        # Y = Y + mult_add(Y, bias[i])
        # print(Y.shape, W[i].shape)
        # Y = cp.cusparse.csrgemm2(a=Y, b=W[i])
        
        # print(cp.where(Y.data < 0))
        # Threshold negative values.
        # Y.data[cp.where(Y.data < 0)] = 0
        # Y.data = cp.where(Y.data < 0, 0, Y.data)
        
        # Threshold maximum values.
        # Y.data[cp.where(Y.data > YMAX)] = YMAX
        # Y.data = cp.where(Y.data > YMAX, YMAX, Y.data)
        Y.data = cp.where(Y.data <- bias,  0, cp.where(Y.data > YMAX - bias, YMAX, Y.data + bias))
        
        # update_kern(((Y.data.size//1024)+1,), (1024,), (Y.data, YMAX))
        Y.eliminate_zeros()
        # print(Y.size)
        # iteration_times.append(time.perf_counter()-tic)

    # print(cp.mean(cp.array(iteration_times)))
    return Y