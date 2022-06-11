import sys
import cupy as cp
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = Y0

    # loop through each weight layer W[i]
    # print(Y.size)
    for i in range(len(W)):
        
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        # print(W[i].shape)
        # print(Y.shape)
        Z = Y * W[i]
        b = bias[i]
        # print(Z.shape)
        # print(b.T.shape)
        # print(Z.multiply(b).shape)
        # sys.exit(b.shape)

        # Apply bias to non-zero entries.
        # !!!! COULD BE MADE EFFICIENT
        Y = Z + (Z.astype('bool').astype('float').multiply(b))
        # sys.exit()
        
        # Y_zeros = Y < 0
        # Y_indptr = Y_zeros.indptr
        # Y_indices = Y_zeros.indices
        # Y_rows = cp.zeros_like(Y_indices)
        # for i in range(Y_indptr.shape[0]-1):
        #     Y_rows[Y_indptr[i]:Y_indptr[i+1]] = i
        # print(Y_indptr[0])

        # Threshold negative values.
        # i, j = (Y<0).nonzero()
        # print(i, j)
        # print(Y.nonzero())
        # print(Y.data.shape)
        # print(.row)
        # print((Y<0))
        # print((Y<0).indices)
        # print((Y<0).indptr)
        # print((Y<0).data)
        Y.data[cp.where(Y.data < 0)] = 0
        # print(Y.size)
        # print(Y.size)
        
        # sys.exit(Y)
        # indptr

        # Threshold maximum values.
        Y.data[cp.where(Y.data > YMAX)] = YMAX
        Y.eliminate_zeros()

    # print(Y.size)
    # Y.eliminate_zeros()
    # print(Y.size)
    return Y