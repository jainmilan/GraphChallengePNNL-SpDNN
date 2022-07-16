for n in 1024 4096 16384 65536
do
    for l in 120 480 1920
    do 
        # echo "$n $l"
        sh run_sbatch.sh ${1} ${2} 1 cupy_horovod $n $l 2 0
    done
done