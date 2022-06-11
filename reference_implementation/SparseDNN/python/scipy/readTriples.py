from scipy.sparse import csr_matrix
import pandas as pd

def readTriples(fname):
    # Read triples for a matrix from a TSV file and
    # build a sparse matrix from the triples.

    # Read data from file into a triples matrix.
    # ijv = transpose(reshape(sscanf(StrFileRead(fname), '%f'), 3, []));
    ijv = StrFileRead(fname)
    # print(ijv)
    # print(ijv[0].values)

    # Create sparse matrix from triplses.
    A = csr_matrix((ijv[2].values, (ijv[0].values, ijv[1].values)));
    # print(A)
    # , shape=(60000, 1024)
    
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
    df[0] = df[0] - 1
    df[1] = df[1] - 1
    
    # df.loc[df.shape[0]] = [60000, 1024, 0]
    # print(df[0].unique(), df[1].unique())
    
    return df