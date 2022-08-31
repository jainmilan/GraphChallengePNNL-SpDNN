import time, sys
import scipy, numpy
import argparse
import cupy as cp
from cupy.sparse import csr_matrix

from readTriples import readTriples, readDense
from inferenceReLUvec import inferenceReLUvec
print("======Versions=======\n Python: %s\n CuPy: %s\n scipy: %s\n numpy: %s" %(sys.version, cp.__version__, scipy.__version__, numpy.__version__))

parser = argparse.ArgumentParser()
parser.add_argument('--neurons', default=1024, choices=[1024, 4096, 16384, 65536], help="Number of neurons for training [options: 1024, 4096, 16384, 65536], defaults to 1024.", type=int)

args = parser.parse_args()

# Set locations of files.
basePath = '/qfs/projects/pacer/milan/data/SparseDNNChallenge/'

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
maxLayers = [120];

# Set DNN bias.
neuralNetBias = [-0.3,-0.35,-0.4,-0.45];

# Loop over each DNN.
for i in range (len(Nneuron)):
    # Load sparse MNIST data.
    if READTSV:
        # filename = inputFile + str(Nneuron[i]) + '.tsv'
        filename = f"{inputFile}{Nneuron[i]}.tsv"
        print("[INFO] Reading file: %s" %(filename))
        featureVectors = readTriples(filename, n_features=Nneuron[i])
    
    if READMAT:
        # filename = 'data/MNIST/sparse-images-' + str(Nneuron(i)) + '.mat'
        # scipy.io.loadmat(filename, 'z')
        # featureVectors = z
        pass

    NfeatureVectors = featureVectors.shape[0]
    
# Read layers.
for j in range(len(maxLayers)):
    # Read in true categories.
    if not SAVECAT:
        filename = f"{categoryFile}{Nneuron[i]}-l{maxLayers[j]}-categories.tsv"
        # print(filename)
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
            # print(filename)
            layers.append(readDense(filename, n_features=Nneuron[i], order='f'));
        if READMAT:
            # load([layerFile num2str(Nneuron[i]) '/n' num2str(Nneuron(i)) '-l' num2str(k) '.mat'],'layersScaledj');
            # layers{k} = layersScaledj;
            pass
        DNNedges = DNNedges + cp.count_nonzero(layers[k]);
        bias.append((csr_matrix(cp.ones((1, Nneuron[i]))) * neuralNetBias[i]).todense())
    
    readLayerTime = time.perf_counter() - tic
    readLayerRate = DNNedges/readLayerTime;

    print('[INFO] DNN neurons/layer: %d, layers: %d, edges: %d' %(Nneuron[i], maxLayers[j], DNNedges))
    print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate));

    # Perform and time challenge
    tic = time.perf_counter();
    scores = inferenceReLUvec(layers, bias, featureVectors);  
    # scores = featureVectors
    challengeRunTime = time.perf_counter() - tic;

    challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime;
    print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate));

    # Compute categories from scores.
    scores_sum = scores.sum(axis=1)
    categories, col = scores_sum.nonzero()
    val = scores_sum
    
    if SAVECAT:
    #   StrFileWrite(sprintf('%d\n',categories),[categoryFile num2str(Nneuron(i)) '-l' num2str(maxLayers(j)) '-categories.tsv']);
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Software Engineer: Dr. Jeremy Kepner                    
# % MIT                   
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % (c) <2019> Massachusetts Institute of Technology
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

