import time, sys
import scipy
import argparse
import cupy as cp
import numpy as np
import pandas as pd
from cupy.sparse import csr_matrix
from scipy.sparse import csr_matrix as scipy_csr_matrix

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from readTriples import readTriples
from inferenceReLUvec import inferenceReLUvec
if rank == 0: 
    print("======Versions=======\n Python: %s\n CuPy: %s\n scipy: %s\n numpy: %s\n pandas: %s" %(sys.version, cp.__version__, scipy.__version__, np.__version__, pd.__version__))

parser = argparse.ArgumentParser()
parser.add_argument('--neurons', default=1024, choices=[1024, 4096, 16384, 65536], help="Number of neurons for training [options: 1024, 4096, 16384, 65536], defaults to 1024.", type=int)
parser.add_argument('--num_layers', default=120, choices=[120, 480, 1920], help="Number of neurons for training [options: 120, 480, 1920], defaults to 120.", type=int)

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
if rank == 0:
    print("[INFO] Neural Net Bias: %f" %(neuralNetBias_val))

# neuralNetBias = [-0.3,-0.35,-0.4,-0.45];
neuralNetBias=neuralNetBias_val
NfeatureVectors = 60000

# Loop over each DNN.
# if rank == 0:
# Load sparse MNIST data.
filename = f"{inputFile}{Nneuron}.tsv"
if rank == 0:
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
trueCategories = np.genfromtxt(filename)
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

if rank == 0:
    print('[INFO] DNN neurons/layer: %d, layers: %d, edges: %d' %(Nneuron, maxLayers, DNNedges))
    print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate));

# Perform and time challenge
split_number = NfeatureVectors // size
start = rank * split_number
end = start + split_number
print("[INFO] Processing Batch: [%d, %d]" %(start, end))

# layersData = [csr_matrix(l, dtype=cp.float32) for l in layers]
layersData = layers

tic = time.perf_counter();
with cp.cuda.Device(rank):
    scores_batched, spmmTime = inferenceReLUvec(layersData, bias, featureVectors[:, start:end].todense(order='f'))
challengeRunTime = time.perf_counter() - tic;

if rank == 0:
    # Compute categories from scores.
    print("Challenge Time: %f" %(challengeRunTime))
    print("SpMM Time: %f" %(spmmTime))

# challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime;
# Compute categories from scores.
# print(challengeRunTime)
# print(challengeRunRate)
spmm_times = comm.reduce(spmmTime, op=MPI.SUM, root=0)
run_times = comm.reduce(challengeRunTime, op=MPI.SUM, root=0)
scores_batched = comm.gather(scores_batched, root=0)
# run_rates = comm.reduce(challengeRunRate, op=MPI.SUM, root=0)
# if rank == 0:
#     print('[INFO] SpMM time (sec): %f, Run time (sec): %f, run rate (edges/sec): %f' %(spmm_times/size, run_times/size, run_rates/size));

# run_rates = comm.reduce(challengeRunRate, op=MPI.SUM, root=0)
if rank == 0:
    spmm_time = spmm_times / size
    spmm_rate = NfeatureVectors * DNNedges / spmm_time
    iteration_time = run_times / size
    iteration_rate = NfeatureVectors * DNNedges / run_times;
    print('[INFO] SpMM time (sec): %f, SpMM Run rate (edges/sec): %f, Iteration time (sec): %f, Iteration Run rate (edges/sec): %f' %(spmm_time, spmm_rate, iteration_time, iteration_rate));

    # print("Scored Batched", scores_batched, scores_batched[0].shape, scores_batched[1].shape)
    scores = np.hstack(scores_batched)
    # print("Scores", scores, scores.shape)
    scores_sum = cp.asnumpy(scores.sum(axis=0))
    # print("Scores Sum", scores_sum)
    categories = scores_sum.nonzero()[0]
    val = scores_sum
    
    if SAVECAT:
        pass
    else:
        # print(trueCategories, categories)
        # print(np.ones_like(trueCategories).shape, np.zeros_like(trueCategories).shape, np.ones_like(categories).shape, np.zeros_like(categories).shape)
        tc_sparse = scipy_csr_matrix((np.ones_like(trueCategories), (np.array(trueCategories), np.zeros_like(trueCategories))), shape=(NfeatureVectors, 1), dtype='float32')
        pc_sparse = scipy_csr_matrix((np.ones_like(categories), (np.array(categories), np.zeros_like(categories))), shape=(NfeatureVectors, 1), dtype='float32')
        categoryDiff = tc_sparse - pc_sparse
        print("[INFO] Non-zero category difference: %d" %(categoryDiff.count_nonzero()))
        if (categoryDiff.count_nonzero()):
            print('[INFO] Challenge FAILED');
        else:
            print('[INFO] Challenge PASSED');    

