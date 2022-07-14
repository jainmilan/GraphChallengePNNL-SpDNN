import argparse
import numpy as np
from scipy.sparse import csr_matrix

basePath = '/qfs/projects/pacer/graphchallenge2022/'
categoryFile = basePath + 'DNN/neuron';


parser = argparse.ArgumentParser(
    description='Validate the scores'
)
parser.add_argument(
	"--score_file", 
	help="Score file to validate", 
	required=True
)
parser.add_argument(
	"--neurons", 
	help="Number of neurons", 
	default=1024
)
parser.add_argument(
	"--max_layers", 
	help="Maximum number of layers", 
	default=120
)

if __name__ == "__main__":
    args = parser.parse_args()
    scoreFile = args.score_file

    # scores
    scoresSum = np.loadtxt(scoreFile)
    categories = np.nonzero(scoresSum)[0]
    print("[INFO] Predicted Categories Shape:", categories.shape)

    # true categories
    filename = f"{categoryFile}{args.neurons}-l{args.max_layers}-categories.tsv"
    trueCategories = np.genfromtxt(filename)
    # FIXING THE INDEXING: True Categories are +1
    trueCategories = trueCategories - 1
    print("[INFO] True Categories Shape:", trueCategories.shape)

    tc_sparse = csr_matrix((np.ones_like(trueCategories), (trueCategories, np.zeros_like(trueCategories))), shape=(60000, 1), dtype='float32')
    pc_sparse = csr_matrix((np.ones_like(categories), (categories, np.zeros_like(categories))), shape=(60000, 1), dtype='float32')
    categoryDiff = tc_sparse - pc_sparse
    print("[INFO] Non-zero category difference: %d" %(categoryDiff.count_nonzero()))
    if (categoryDiff.count_nonzero()):
        print('[INFO] Challenge FAILED');
    else:
        print('[INFO] Challenge PASSED');