from scipy.sparse import csr_matrix
import pandas as pd

def readTriples(fname):
    # Read triples for a matrix from a TSV file and
    # build a sparse matrix from the triples.

    # Read data from file into a triples matrix.
    # ijv = transpose(reshape(sscanf(StrFileRead(fname), '%f'), 3, []));
    ijv = StrFileRead(fname)
    # print(ijv)

    # Create sparse matrix from triplses.
    A = csr_matrix((ijv[:,2], (ijv[:,0], ijv[:,1])));
    
    return A

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
    # print(df)
    
    return df.values