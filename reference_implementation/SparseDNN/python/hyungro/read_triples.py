import numpy as np
import pandas as pd
from cupyx.scipy import sparse
import scipy.sparse
import cupy as cp

def read_input(fname):

    ijv = pd.read_csv(fname, sep='\t', header=None)
    # index starting from 0
    ijv[0] -= 1
    ijv[1] -= 1

    neuron = 1024
    if ijv.shape[0] == 25019051:
        neuron = 4096
    elif ijv.shape[0] == 98858913:
        neuron = 16384
    elif ijv.shape[0] == 392191985:
        neuron = 65536

    data = cp.array(ijv[2].values)
    row = cp.array(ijv[0].values)
    col = cp.array(ijv[1].values)

    A = sparse.csr_matrix((data, (row, col)), shape=(60000, neuron), dtype=cp.float32)
    
    return A.T.todense(order='f')

def read_weight(fname):
    ijv = pd.read_csv(fname, sep='\t', header=None)
    # index starting from 0
    ijv[0] -= 1
    ijv[1] -= 1

    data = cp.array(ijv[2].values)
    row = cp.array(ijv[0].values)
    col = cp.array(ijv[1].values)

    A = sparse.csr_matrix((data, (row, col)), dtype='float32')
    return A.T
