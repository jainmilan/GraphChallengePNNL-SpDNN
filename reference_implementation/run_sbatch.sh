#!/bin/sh

echo "--------- Benchmarking ReLU -------------------"
echo "Partition: ${1}"
echo "Number of GPUs: ${2}"
echo "Number of CPUs: ${3}"
echo "Version: ${4}"
echo "Neurons: ${5}"
echo "Number of Layers: ${6}"
echo "Profiling: ${7}"

if [ ${8} == 0 ]; then
    sbatch -A pacer -N 1 -t 02:00:00 -p ${1} --gres=gpu:${2} --reservation=pacer -J GC_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6} -o ../../../logs/GraphChallenge/sbatch/out_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6}.txt -e ../../../logs/GraphChallenge/sbatch/err_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6}.txt run.sh ${1} ${2} ${3} ${4} ${5} ${6} ${7}
elif [ ${8} == 1 ]; then
    sbatch -A pacer -N 1 -t 02:00:00 -p ${1} --gres=gpu:${2} --constraint=nvlink -J GC_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6} -o ../../../logs/GraphChallenge/sbatch/out_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6}.txt -e ../../../logs/GraphChallenge/sbatch/err_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6}.txt run.sh ${1} ${2} ${3} ${4} ${5} ${6} ${7}
elif [ ${8} == 2 ]; then
    sbatch -A pacer -N 1 -t 02:00:00 -p ${1} --gres=gpu:${2} -J GC_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6} -o ../../../logs/GraphChallenge/sbatch/out_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6}.txt -e ../../../logs/GraphChallenge/sbatch/err_p${1}_ng${2}_nc${3}_vng${4}_n${5}_nl${6}.txt run.sh ${1} ${2} ${3} ${4} ${5} ${6} ${7}
elif [ ${8} == 5 ]; then
    sbatch -A pacer -N 1 -t 02:00:00 -p ${1} --gres=gpu:1 -J GC_p${1}_ng${2}_demo -o ../../../logs/GraphChallenge/sbatch/out_p${1}_ng${2}_demo.txt -e ../../../logs/GraphChallenge/sbatch/err_p${1}_ng${2}_demo.txt run.sh ${1} ${2} ${3} ${4} ${5} ${6} ${8} ${8}
fi