#!/bin/sh

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

export TEMP=/qfs/projects/pacer/milan/scratch/temp/
export TMP=/qfs/projects/pacer/milan/scratch/tmp/
export TEMPDIR=/qfs/projects/pacer/milan/scratch/tempdir/
export TMPDIR=/qfs/projects/pacer/milan/scratch/tmpdir/

echo "--------- Benchmarking ReLU -------------------"
echo "Partition: ${1}"
echo "Number of GPUs: ${2}"
echo "Number of CPUs: ${3}"
echo "Version: ${4}"
echo "Neurons: ${5}"

module load cuda/11.4
ulimit -u 16000
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+"$LD_LIBRARY_PATH:"}/share/apps/cuda/11.4/targets/x86_64-linux/lib/"
if [ -d "/share/apps/cuda/11.4/extras/CUPTI/lib64/" ] && [[ ":$LD_LIBRARY_PATH:" != *":/share/apps/cuda/11.4/extras/CUPTI/lib64/:"* ]]; then
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+"$LD_LIBRARY_PATH:"}/share/apps/cuda/11.4/extras/CUPTI/lib64/"
fi
echo "LD Library Path: ${LD_LIBRARY_PATH}"
conda activate horovod

python --version
# echo ${PWD}/SparseDNN/python/scipy/inferenceReLUvec.py
python -u ${PWD}/SparseDNN/python/${4}/runSparseDNNchallenge.py --neurons ${5}

