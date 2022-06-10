import scipy

# Set locations of files.
inputFile = '../../../../../data/SparseDNNChallenge/MNIST/sparse-images-';
categoryFile = '../../../../../data/SparseDNNChallenge/DNN/neuron';
layerFile = '../../../../../data/SparseDNNChallenge/DNN/neuron';

# Select DNN to run.
#Nneuron = [1024, 4096, 16384, 65536];
Nneuron = [1024];
SAVECAT = False    # Overwrite truth categories.
READTSV = True    # Read input and layers from .tsv files.
READMAT = True    # Redd input and layers from .mat files.

# Select number of layers to run.
#maxLayers = 120 * [1, 4, 16];
maxLayers = [120];

# Set DNN bias.
neuralNetBias = [-0.3,-0.35,-0.4,-0.45];

from readTriples import readTriples

# Loop over each DNN.
for i in range (len(Nneuron)):
    # Load sparse MNIST data.
    if READTSV:
        filename = inputFile + str(Nneuron[i]) + '.tsv'
        print("[INFO] Reading file: %s" %(filename))
        featureVectors = readTriples(filename)
    
    # if READMAT:
    #     filename = 'data/MNIST/sparse-images-' + str(Nneuron(i)) + '.mat'
    #     scipy.io.loadmat(filename, 'z')
    #     featureVectors = z
    featureVectors[-1, Nneuron(i)] = 0       # Pad matrix.
    NfeatureVectors = featureVectors[0]
    print(featureVectors, NfeatureVectors)
  
#   % Read layers.
#   for j=1:length(maxLayers)
  
#     % Read in true categories.
#     if not(SAVECAT)
#       trueCategories = str2num(StrFileRead([categoryFile num2str(Nneuron(i)) '-l' num2str(maxLayers(j)) '-categories.tsv']));
#     end

#     DNNedges = 0;
#     layers = {};
#     bias = {};
#     tic;
#       for k=1:maxLayers(j)
#         if READTSV
#           layers{k} = readTriples([layerFile num2str(Nneuron(i)) '/n' num2str(Nneuron(i)) '-l' num2str(k) '.tsv']);
#         end
#         if READMAT
#           load([layerFile num2str(Nneuron(i)) '/n' num2str(Nneuron(i)) '-l' num2str(k) '.mat'],'layersScaledj');
#           layers{k} = layersScaledj;
#         end
#         DNNedges = DNNedges + nnz(layers{k});
#         bias{k} = sparse(ones(1,Nneuron(i)) .* neuralNetBias(i));
#       end
#     readLayerTime = toc;
#     readLayerRate = DNNedges/readLayerTime;

#     disp(['DNN neurons/layer: ' num2str(Nneuron(i)) ', layers: ' num2str(maxLayers(j)) ', edges: ' num2str(DNNedges)]);
#     disp(['Read time (sec): ' num2str(readLayerTime) ', read rate (edges/sec): ' num2str(readLayerRate)]);

#     % Perform and time challenge
#     tic;
#        scores = inferenceReLUvec(layers,bias,featureVectors);  
#     challengeRunTime = toc;

#     challengeRunRate = NfeatureVectors*DNNedges/challengeRunTime;
#     disp(['Run time (sec): ' num2str(challengeRunTime) ', run rate (edges/sec): ' num2str(challengeRunRate)]);

#     % Compute categories from scores.
#     [categories col val] = find(sum(scores,2));

#     if SAVECAT
#       StrFileWrite(sprintf('%d\n',categories),[categoryFile num2str(Nneuron(i)) '-l' num2str(maxLayers(j)) '-categories.tsv']);
#     else
#       categoryDiff = sparse(trueCategories,1,1,NfeatureVectors,1) - sparse(categories,1,1,NfeatureVectors,1);
#       if (nnz(categoryDiff))
#         disp('Challenge FAILED');
#       else
#         disp('Challenge PASSED');
#       end
#     end
    
#   end

# end


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Software Engineer: Dr. Jeremy Kepner                    
# % MIT                   
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % (c) <2019> Massachusetts Institute of Technology
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

