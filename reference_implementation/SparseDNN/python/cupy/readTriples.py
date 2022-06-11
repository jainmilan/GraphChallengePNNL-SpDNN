import cupy as cp
from cupy.sparse import csr_matrix
import pandas as pd
# import cudf

def readTriples(fname, n_features):
    # Read triples for a matrix from a TSV file and
    # build a sparse matrix from the triples.

    # Read data from file into a triples matrix.
    # ijv = transpose(reshape(sscanf(StrFileRead(fname), '%f'), 3, []));
    ijv = StrFileRead(fname)
    
    A = csr_matrix((ijv[:, 2], (ijv[:, 0], ijv[:, 1])));
    B = csr_matrix((ijv[:, 2], (ijv[:, 0], ijv[:, 1])), shape=(A.shape[0], n_features));
    
    return B

def StrFileRead(file):
    #StrFileRead: Reads a file into a string array.
    #String utility function.
    #  Usage:
    #    s = StrFileRead(file)
    #  Inputs:
    #    file = filename
    #  Outputs:
    #    s = string
    
    df = pd.read_csv(file, delimiter='\t', header=None)
    df[0] = df[0] - 1
    df[1] = df[1] - 1
    
    return cp.asarray(df.values, dtype='float32')