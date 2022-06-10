import numpy as np

def inferenceReLUvec(W, bias, Y0):
    
    YMAX = 32 # set max value

    Y = Y0

    # loop through each weight layer W[i]
    for i in range(len(W)):
        
        # % Propagate through layer.
        # % Note: using graph convention of A(i,j) means connection from i *to* j,
        # % that requires *left* multiplication feature *row* vectors.
        Z = Y * W[i]
        b = bias[i]

        # Apply bias to non-zero entries.
        Y = Z + (np.array(np.full((Z), True), dtype=float) * b)

        # Threshold negative values.
        Y[Y < 0] = 0

        # Threshold maximum values.
        Y[Y > YMAX] = YMAX

    return Y
