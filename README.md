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

The official data sets are linked from the Graph Challenge website: http://graphchallenge.mit.edu/data-sets

## Requirement

- CuPy v10.6.0
- CUDA 11.4+
- gcc 7.5.0+

## Run

### Python via CuPy `csr2gemm` and `spmm`

- Adjust `base_path` in the `runSparseDNNchallenge.py` main script
- example of running 1024 neurons and 120 layers: `python runSparseDNNchallenge.py --neurons 1024 --num_layers 120`

### OpenMP

TBD

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
