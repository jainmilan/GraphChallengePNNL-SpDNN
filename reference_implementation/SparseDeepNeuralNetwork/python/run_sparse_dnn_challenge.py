from read_triples import readTriples
from inference import inferenceReLUvec
import numpy as np
from scipy import sparse as sps
import time

base_path = '/qfs/projects/pacer/graphchallenge2022/'

inputFile = './MNIST/sparse-images-'
categoryFile = './DNN/neuron'
layerFile = './DNN/neuron'

Nneuron = [1024] # [1024, 4096, 16384, 65536]
SAVECAT = 0
READTSV = 1
READMAT = 0

maxLayers = [120] # * [ 1, 4, 16]

neuralNetBias = [-0.3, -0.35, -0.4, -0.45]

for i in range(len(Nneuron)):
    if READTSV:
        featureVectors = readTriples(f"{base_path}{inputFile}{Nneuron[i]}.tsv")
    if READMAT:
        pass

    # pad matrix
    NfeatureVectors = featureVectors.size

    for j in range(len(maxLayers)):
        if not(SAVECAT):
            trueCategories = np.genfromtxt(f"{base_path}{categoryFile}{Nneuron[i]}-l{maxLayers[j]}-categories.tsv")

        DNNedges = 0
        layers = {}
        bias = {}
        tic = time.time()
        for k in range(1, maxLayers[j]):
            if READTSV:
                layers[k] = readTriples(f"{base_path}{layerFile}{Nneuron[i]}/n{Nneuron[i]}-l{k}.tsv")
            if READMAT:
                pass
        print(k)
        if k == 119:
            print(layers[k])
        DNNedges = DNNedges + np.count_nonzero(layers[k])
        bias[k] = sps.csr_matrix(np.ones((1, Nneuron[i]))) * neuralNetBias[i]
    readLayerTime = time.time() - tic
    readLayerRate = DNNedges / readLayerTime

    print(f"DNN neurons/layers: {Nneuron[i]}, layers: {maxLayers[j]}, edges: {DNNedges}")
    print(f"Read time (sec): {readLayerTime}, read rate (edges/sec): {readLayerRate}")

    tic = time.time()
    scores = inferenceReLUvec(layers, bias, featureVectors)
    challengeRunTime = time.tine() - tic

    challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime
    print(f"Run time (sec): {challengeRunTime}, run rate (edges/sec): {challengeRunRate}")

    # Compute categories from scores
    categories, col, val = np.where(sum(scores, 2))

    if SAVECAT:
        pass
    else:
        categoryDiff = sps.csr_matrix(trueCategories, 1, 1, NfeatureVectors, 1) - sps.csr_matrix(categories, 1, 1, NfeatureVectors, 1)
        if np.count_nonzero(categoriDiff):
            print ('Challenge FAILED')
        else:
            print ('Challenge PASSED')

