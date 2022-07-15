import time, sys
import scipy, numpy
import argparse
import cupy as cp
import numpy as np
from cupy.sparse import csr_matrix

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from readTriples import readTriples
from inferenceReLUvec import inferenceReLUvec
if rank == 0: 
    print("======Versions=======\n Python: %s\n CuPy: %s\n scipy: %s\n numpy: %s" %(sys.version, cp.__version__, scipy.__version__, numpy.__version__))

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

# Loop over each DNN.
if rank == 0:
    # Load sparse MNIST data.
    filename = f"{inputFile}{Nneuron}.tsv"
    if rank == 0:
        print("[INFO] Reading file: %s" %(filename))
    featureVectors = readTriples(
        filename, 
        n_rows=60000,
        n_features=Nneuron, 
        subtract=subtract_val
    )

NfeatureVectors = 60000
    
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

if rank == 0:
    print('[INFO] DNN neurons/layer: %d, layers: %d, edges: %d' %(Nneuron, maxLayers, DNNedges))
    print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate));

# Perform and time challenge
split_number = 60000//size
start = rank * split_number
end = start + split_number
print("[INFO] Processing Batch: [%d, %d]" %(start, end))

if rank == 0:
    featureData = featureVectors[start:end]
    for device_num in range(1, size):
        start_local = device_num * split_number
        end_local = start_local + split_number
        comm.send(featureVectors[start_local:end_local], dest=device_num)
else:
    featureData = comm.recv(source=0)

tic = time.perf_counter();
with cp.cuda.Device(rank):
    scores_batched = inferenceReLUvec(layers, bias, featureData)

challengeRunTime = time.perf_counter() - tic;

challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime;

# Compute categories from scores.
print(challengeRunTime)
# print(challengeRunRate)
run_times = comm.reduce(challengeRunTime, op=MPI.SUM, root=0)
run_rates = comm.reduce(challengeRunRate, op=MPI.SUM, root=0)
if rank == 0:
    print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(run_times/size, run_rates/size));

scores_batched = comm.gather(scores_batched, root=0)
if rank==0:
    scores = cp.sparse.vstack(scores_batched)
    scores_sum = scores.sum(axis=1)
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

