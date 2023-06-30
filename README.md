# SpDNN Inference Using Different Programming Models

This github repository is the codebase of the GraphChallengePNNL 2022 using Python and C/C++ programming models.

CuPy, CUDA cuSPARSE, and OpenMP are included in the context of Sparse Deep Neural Network (SpDNN) implementations (derived from the Graph Challenge reference serial code in MATLAB) and the performance results were produced using single and multiple GPUs from NVIDIA DGX-A100 40GB/80GB platforms.

# Quick Start

We use sub-directories for different programming models with variations e.g., csr2gemm and spmm.


- reference_implementation/SparseDNN/matlab: matlab base code
- reference_implementation/SparseDNN/cusparse: c++
- reference_implementation/SparseDNN/openmp: openmp offload
- reference_implementation/SparseDNN/openmp-batch: openmp batch
- reference_implementation/SparseDNN/python/cupy: CuPy csr2gemm
- reference_implementation/SparseDNN/python/cupy_spmm: CuPy spmm


## Getting Data sets

The official data sets are provided in text files (.tsv) for MNIST digit images. For the purpose of fast-reading the inputs (with a smaller file size in an aggregated single file), we converted them to binary formats across all graph images (from 1024 to 65536 neurons). 

The conversion takes simple processes a) reading each line of files with a template (`row_index\tcolumn_index\t1`) which represent image pixel information in sparse data, b) iterating a) per neuron (weight) to concatenate if possible, and c) writing them into a binary format (`"wb"` mode in `fopen`) with a fixed size of `integer` bytes (for row and column) and `float` (for value) for storing the array of elements (row, column, and 1).

Once the binary conversion is complete, we read the block of data using the same format used in writing, and we use compressed sparse row (CSR) format to build a matrix (in C++ implementation and in Python CuPy sparse extended package `csr_matrix` we use) for the challenge, spDNN calculation.

- The original datasets are also available on Amazon S3, which can be found in the links on the Graph Challenge website: http://graphchallenge.mit.edu/data-sets.

## Requirement

- CuPy v10.6.0
- CUDA 11.4+
- gcc 7.5.0+

## Run

### Python via CuPy `csr2gemm` and `spmm`

- Adjust `base_path` in the `runSparseDNNchallenge.py` main script
- example of running 1024 neurons and 120 layers: `python runSparseDNNchallenge.py --neurons 1024 --num_layers 120`

### OpenMP

The OpenMP version uses a simple SpMM (Sparse Matrix [sparse] Matrix [dense] multiplication) kernel and OpenMP (4.5) offload model to port the kernel on a GPU, instead of using a third-party numerical library like the other variants. The bias computations are performed within the kernel itself. We observed that for NVHPC to work properly for the batched version, `-gpu=managed` option must be passed.

### C++

- Adjust parameters in the code e.g., base_path
- `make clean; make all`
- `./cusparse`


# Citation

```
@inproceedings{tbd,
  title={Sparse Deep Neural Network Inference Using Different Programming Models},
  author={Lee, Hyungro and Jain, Milan and Ghosh, Sayan},
  booktitle={2022 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={},
  year={2022},
  organization={IEEE}
}
```


# License

The repo has [BSD-3-Clause license](LICENSE).
