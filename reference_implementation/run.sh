#!/bin/sh

module purge
module load python/miniconda3.9
module load gcc/9.1.0
module load openmpi/4.1.0
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
echo "Number of Layers: ${6}"
echo "Profiling: ${7}"

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
# python -u ${PWD}/SparseDNN/python/${4}/runSparseDNNchallenge.py --neurons ${5}

if [ ${7} == 1 ]; then
    nsys profile --kill=none -t cuda,osrt,cudnn,cublas -o ../../../logs/GraphChallenge/nsys/qdrep_report_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6} -w true --force-overwrite=true python -u ${PWD}/SparseDNN/python/${4}/runSparseDNNchallenge.py --neurons ${5} --num_layers ${6}
elif [ ${7} == 0 ]; then
    python -u ${PWD}/SparseDNN/python/${4}/runSparseDNNchallenge.py --neurons ${5} --num_layers ${6}
elif [ ${7} == 2 ]; then
    horovodrun -np ${2} python -u ${PWD}/SparseDNN/python/${4}/runSparseDNNchallenge.py --neurons ${5} --num_layers ${6}
elif [ ${7} == 3 ]; then
    nsys profile --kill=none -t cuda,osrt,cudnn,cublas -o ../../../logs/GraphChallenge/nsys/qdrep_report_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6} -w true --force-overwrite=true horovodrun -np ${2} python -u ${PWD}/SparseDNN/python/${4}/runSparseDNNchallenge.py --neurons ${5} --num_layers ${6}
fi
# 

