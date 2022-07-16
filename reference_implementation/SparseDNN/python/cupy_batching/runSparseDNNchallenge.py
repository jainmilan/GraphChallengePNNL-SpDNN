import time, sys
import scipy, numpy
import argparse
import cupy as cp
import numpy as np
from cupy.sparse import csr_matrix

from readTriples import readTriples
from inferenceReLUvec import inferenceReLUvec
print("======Versions=======\n Python: %s\n CuPy: %s\n scipy: %s\n numpy: %s" %(sys.version, cp.__version__, scipy.__version__, numpy.__version__))

parser = argparse.ArgumentParser()
parser.add_argument('--neurons', default=1024, choices=[1024, 4096, 16384, 65536], help="Number of neurons for training [options: 1024, 4096, 16384, 65536], defaults to 1024.", type=int)
parser.add_argument('--num_layers', default=120, choices=[120, 480, 1920], help="Number of neurons for training [options: 120, 480, 1920], defaults to 120.", type=int)
parser.add_argument('--num_batches', default=6, choices=[1, 2, 3, 4, 6], help="Number of neurons for training [options: 120, 480, 1920], defaults to 120.", type=int)

args = parser.parse_args()

# Set locations of files.
# basePath = '/qfs/projects/pacer/milan/data/SparseDNNChallenge/'
basePath = '/qfs/projects/pacer/graphchallenge2022/'

inputFile = basePath + 'MNIST/sparse-images-';
categoryFile = basePath + 'DNN/neuron';
layerFile = basePath + 'DNN/neuron';

# Select DNN to run.
#Nneuron = [1024, 4096, 16384, 65536];
Nneuron = args.neurons
SAVECAT = False    # Overwrite truth categories.
READTSV = True    # Read input and layers from .tsv files.
READMAT = True    # Redd input and layers from .mat files.

# Select number of layers to run.
#maxLayers = 120 * [1, 4, 16];
maxLayers = args.num_layers

# Set DNN bias.
neuralNetBias_val = -0.3
subtract_val = True
if args.neurons == 4096:
    neuralNetBias_val = -0.35
elif args.neurons == 16384:
    neuralNetBias_val = -0.4
elif args.neurons == 65536:
    neuralNetBias_val = -0.45
    # subtract_val = False
print("[INFO] Neural Net Bias: %f" %(neuralNetBias_val))

# neuralNetBias = [-0.3,-0.35,-0.4,-0.45];
neuralNetBias=neuralNetBias_val
NfeatureVectors = 60000

# Load sparse MNIST data.
filename = f"{inputFile}{Nneuron}.tsv"
print("[INFO] Reading file: %s" %(filename))
featureVectors = readTriples(
    filename, 
    n_rows=NfeatureVectors,
    n_features=Nneuron, 
    subtract=subtract_val
)

# Read layers.
# Read in true categories.
filename = f"{categoryFile}{Nneuron}-l{maxLayers}-categories.tsv"
trueCategories = cp.genfromtxt(filename)
# FIXING THE INDEXING: True Categories are +1
trueCategories = trueCategories - 1
    
DNNedges = 0;
layers = [];
bias = [];
tic = time.perf_counter();

# read layers
for k in range(maxLayers):
    filename = f"{layerFile}{Nneuron}/n{Nneuron}-l{k+1}.tsv"
    layers.append(readTriples(
        filename, 
        n_rows=Nneuron,
        n_features=Nneuron
    ));
    
    DNNedges = DNNedges + layers[k].count_nonzero();

# bias value
bias = neuralNetBias

readLayerTime = time.perf_counter() - tic
readLayerRate = DNNedges/readLayerTime;

print('[INFO] DNN neurons/layer: %d, layers: %d, edges: %d' %(Nneuron, maxLayers, DNNedges))
print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate));

# Perform and time challenge
tic = time.perf_counter();
batch_size = NfeatureVectors // args.num_batches
scores_batched = []
for bn in range(args.num_batches):
    start = bn * batch_size
    end = start + batch_size
    scores_batched.append(inferenceReLUvec(layers, bias, featureVectors[start:end]))

challengeRunTime = time.perf_counter() - tic;

challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime;
print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate));

# Compute categories from scores.
scores = cp.sparse.vstack(scores_batched)
scores_sum = scores.sum(axis=1)
# import numpy as np
# np.savetxt("sample_output.txt", scores_sum.get())
categories, col = scores_sum.nonzero()
val = scores_sum

if SAVECAT:
    pass
else:
    tc_sparse = csr_matrix((cp.ones_like(trueCategories), (trueCategories, cp.zeros_like(trueCategories))), shape=(NfeatureVectors, 1), dtype='float32')
    pc_sparse = csr_matrix((cp.ones_like(categories), (categories, cp.zeros_like(categories))), shape=(NfeatureVectors, 1), dtype='float32')
    categoryDiff = tc_sparse - pc_sparse
    print("[INFO] Non-zero category difference: %d" %(categoryDiff.count_nonzero()))
    if (categoryDiff.count_nonzero()):
        print('[INFO] Challenge FAILED');
    else:
        print('[INFO] Challenge PASSED');    

