import sys
import time
import cupy as cp

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