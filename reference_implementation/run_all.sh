for n in 1024 4096 16384 65536
do
    for l in 120 480 1920
    do 
        # echo "$n $l"
        sh run_sbatch.sh a100_80 ${1} 1 cupy_horovod $n $l 2
    done
done