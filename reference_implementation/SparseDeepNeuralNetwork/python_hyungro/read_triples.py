import numpy as np
import pandas as pd
from scipy import sparse as sps

def readTriples(fname):

    ijv = np.transpose(pd.read_csv(fname, sep='\t'))#, dtype={2:np.float16}))
    A = sps.csr_matrix((ijv.iloc[:,2], (ijv.iloc[:,0], ijv.iloc[:,1])))#, dtype=np.float16)
    
    return A
