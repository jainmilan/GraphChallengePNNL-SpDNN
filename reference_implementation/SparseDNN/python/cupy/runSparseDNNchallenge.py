import time, sys
import scipy, numpy
import argparse
import cupy as cp
from cupy.sparse import csr_matrix

from readTriples import readTriples
from inferenceReLUvec import inferenceReLUvec
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
Nneuron = [args.neurons];
SAVECAT = False    # Overwrite truth categories.
READTSV = True    # Read input and layers from .tsv files.
READMAT = True    # Redd input and layers from .mat files.

# Select number of layers to run.
#maxLayers = 120 * [1, 4, 16];
maxLayers = [args.num_layers];

# Set DNN bias.
neuralNetBias_val = -0.3
if args.neurons == 4096:
    neuralNetBias_val = -0.35
elif args.neurons == 16384:
    neuralNetBias_val = -0.4
elif args.neurons == 65536:
    neuralNetBias_val = -0.45
print("[INFO] Neural Net Bias: %f" %(neuralNetBias_val))

# neuralNetBias = [-0.3,-0.35,-0.4,-0.45];
neuralNetBias=[neuralNetBias_val]

# Loop over each DNN.
for i in range (len(Nneuron)):
    # Load sparse MNIST data.
    if READTSV:
        filename = f"{inputFile}{Nneuron[i]}.tsv"
        print("[INFO] Reading file: %s" %(filename))
        featureVectors = readTriples(filename, n_features=Nneuron[i])
    
    if READMAT:
        pass

    NfeatureVectors = featureVectors.shape[0]
    
    # Read layers.
    for j in range(len(maxLayers)):
        # Read in true categories.
        if not SAVECAT:
            filename = f"{categoryFile}{Nneuron[i]}-l{maxLayers[j]}-categories.tsv"
            trueCategories = cp.genfromtxt(filename)
            # FIXING THE INDEXING: True Categories are +1
            trueCategories = trueCategories - 1
            
        DNNedges = 0;
        layers = [];
        bias = [];
        tic = time.perf_counter();
        for k in range(maxLayers[j]):
            if READTSV:
                filename = f"{layerFile}{Nneuron[i]}/n{Nneuron[i]}-l{k+1}.tsv"
                layers.append(readTriples(filename, n_features=Nneuron[i]));
            if READMAT:
                pass
            DNNedges = DNNedges + layers[k].count_nonzero();
            bias = neuralNetBias[i]
        
        readLayerTime = time.perf_counter() - tic
        readLayerRate = DNNedges/readLayerTime;

        print('[INFO] DNN neurons/layer: %d, layers: %d, edges: %d' %(Nneuron[i], maxLayers[j], DNNedges))
        print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate));

        # Perform and time challenge
        tic = time.perf_counter();
        scores = inferenceReLUvec(layers, bias, featureVectors);  
        challengeRunTime = time.perf_counter() - tic;

        challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime;
        print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate));

        # Compute categories from scores.
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

