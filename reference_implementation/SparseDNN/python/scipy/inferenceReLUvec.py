import numpy as np
import sys
    
def inferenceReLUvec(W, bias, Y0):
    # Performs ReLU inference  using input feature vector(s) Y0,
    # DNN weights W, and constant bias.
    YMAX = 32 # set max value

    # Initialized feature vectors.
    Y = Y0

    # loop through each weight layer W[i]
    # print(W)
    for i in range(len(W)):
        
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        Z = Y * W[i]
        b = bias[i]
        
        # Apply bias to non-zero entries.
        # !!!! COULD BE MADE EFFICIENT
        Y = Z + (Z.astype('bool').astype('float').multiply(b))
        
        # Threshold negative values.
        Y[Y < 0] = 0

        # Threshold maximum values.
        Y[Y > YMAX] = YMAX

    return Y