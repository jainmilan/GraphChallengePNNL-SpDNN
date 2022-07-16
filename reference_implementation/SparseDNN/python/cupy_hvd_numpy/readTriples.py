# import cupy as cp
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
# import cudf

def readTriples(fname, n_rows, n_features, subtract=True):
    # Read triples for a matrix from a TSV file and
    # build a sparse matrix from the triples.

    # Read data from file into a triples matrix.
    # ijv = transpose(reshape(sscanf(StrFileRead(fname), '%f'), 3, []));
    ijv = StrFileRead(fname, subtract=subtract)
    
    # A = csr_matrix((ijv[:, 2], (ijv[:, 0], ijv[:, 1])));
    # print(A.shape)
    B = csr_matrix((ijv[:, 2], (ijv[:, 0], ijv[:, 1])), shape=(n_rows, n_features));
    # print(B.shape)
    
    return B

def StrFileRead(file, subtract=True):
    #StrFileRead: Reads a file into a string array.
    #String utility function.
    #  Usage:
    #    s = StrFileRead(file)
    #  Inputs:
    #    file = filename
    #  Outputs:
    #    s = string
    
    df = pd.read_csv(file, sep='\t', header=None)
    if subtract:
        df[0] = df[0] - 1
        df[1] = df[1] - 1
    
    # return cp.asarray(df.values, dtype=cp.float32)
    return df.values.astype(np.float32)