import cupy as cp
from cupyx.scipy import sparse

# @profile
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = Y0

    # loop through each weight layer W[i]
    for i in range(len(W)):
        
        # tic = time.perf_counter()
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        Y = cp.cusparse.spmm(a=Y, b=W[i])
        
        # Apply bias to non-zero entries.
        Y = cp.where(Y <- bias,  0, cp.where(Y > YMAX - bias, YMAX, Y + bias))
        rows, cols = cp.nonzero(Y)
        
        # eliminate zero entries
        Y = sparse.csr_matrix((Y[rows, cols], (rows, cols)), shape=(60000, 1024))
        
    return Y