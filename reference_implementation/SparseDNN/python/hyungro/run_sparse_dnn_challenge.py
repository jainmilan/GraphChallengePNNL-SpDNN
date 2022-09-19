from read_triples import read_input, read_weight
from inference import inferenceReLUvec
from cupyx.scipy import sparse
import cupy as cp
import numpy as np
import time
import argparse

base_path = '/qfs/projects/pacer/graphchallenge2022/'
#base_path = '/lus/grand/projects/GRACE/spdnn/'

inputFile = './MNIST/sparse-images-'
categoryFile = './DNN/neuron'
layerFile = './DNN/neuron'

Nneuron = [1024 * 1] # [1024, 4096, 16384, 65536]
SAVECAT = 0
READTSV = 1
READMAT = 0
maxLayers = [120]
device = 0
batch_size = 1

parser = argparse.ArgumentParser("spDNN")
parser.add_argument("-n", "--neuron", type=int)
parser.add_argument("-l", "--layer", type=int)
parser.add_argument("-d", "--device", type=int)
parser.add_argument("-b", "--batch_size", type=int)
args = parser.parse_args()

if args.neuron:
    Nneuron = [int(args.neuron)]
if args.layer:
    maxLayers = [int(args.layer)]
if args.device:
    device = int(args.device)
if args.batch_size:
    batch_size = int(args.batch_size)

cp.cuda.Device(device).use()

#print ("[device:{}] {},{}".format( device, Nneuron, maxLayers))
for i in range(len(Nneuron)):
    if READTSV:
        featureVectors = read_input(f"{base_path}{inputFile}{Nneuron[i]}.tsv")
    if READMAT:
        pass

    NfeatureVectors = featureVectors.shape[1]

    for j in range(len(maxLayers)):
        if not(SAVECAT):
            trueCategories = \
                    cp.genfromtxt(f"{base_path}{categoryFile}{Nneuron[i]}"\
                    f"-l{maxLayers[j]}-categories.tsv")

            # index adjustment
            trueCategories -= 1

        DNNedges = 0
        layers = [] 
        #bias = []
        tic = time.time()
        for k in range(maxLayers[j]):
            if READTSV:
                layers.append(read_weight(f"{base_path}{layerFile}{Nneuron[i]}/"\
                        f"n{Nneuron[i]}-l{k+1}.tsv"))
            if READMAT:
                pass
            DNNedges += layers[k].nnz
            #bias.append(cp.multiply(cp.ones((Nneuron[i], 1)), neuralNetBias[i]))
    readLayerTime = time.time() - tic
    readLayerRate = DNNedges / readLayerTime

    print(f"DNN neurons/layers: {Nneuron[i]}, layers: {maxLayers[j]}, edges:"\
            f"{DNNedges}")

    print(f"Read time (sec): {readLayerTime}, read rate (edges/sec): "\
            f"{readLayerRate}")

    spmm_time = [0]
    tic = time.time()
    scores = inferenceReLUvec(layers, featureVectors, spmm_time, batch_size)
    challengeRunTime = time.time() - tic

    challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime
    print("Run time (sec): spmm time:{:.2f}s, iteration time:{:.2f}s, run rate "\
            "(edges/sec): {}, {}".format(spmm_time[0], challengeRunTime,
                NfeatureVectors * DNNedges / spmm_time[0], NfeatureVectors *
                DNNedges), flush=True)

    # Compute categories from scores
    scores_sum = scores.sum(axis=0)
    categories = scores_sum.nonzero()[0]
    val = scores_sum

    if SAVECAT:
        pass
    else:
        categoryDiff = sparse.csr_matrix((cp.ones_like(trueCategories),
            (trueCategories, cp.zeros_like(trueCategories))),
            shape=(NfeatureVectors, 1), dtype='float32') - \
                sparse.csr_matrix((cp.ones_like(categories), (categories,
                    cp.zeros_like(categories))), shape=(NfeatureVectors, 1),
                    dtype='float32')

        if (categoryDiff.count_nonzero()):
            print ('Challenge FAILED')
            print(categoryDiff.count_nonzero())
        else:
            print ('Challenge PASSED')

