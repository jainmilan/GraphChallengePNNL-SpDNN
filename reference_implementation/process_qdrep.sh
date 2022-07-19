#!/bin/bash

#SBATCH -A pacer
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:0
#SBATCH -J process_qdrep
#SBATCH -o oresnet_train.txt
#SBATCH -e eresnet_train.txt
#SBATCH -p slurm

module load cuda/11.4

i=0
for file in /qfs/projects/pacer/milan/logs/GraphChallenge/nsys/*.qdrep
do
    i=$(( i + 1 ))
    name=${file##*/}
    echo "$file"
    if [ "$i" -gt 0 ]; then
        #echo "helloooooooooooooooo"
        #echo ""
        nsys stats -f csv -o . --force-overwrite true $file
    fi
done