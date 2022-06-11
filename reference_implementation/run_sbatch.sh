echo "--------- Benchmarking ReLU -------------------"
echo "Partition: ${1}"
echo "Number of GPUs: ${2}"
echo "Number of CPUs: ${3}"
echo "Version: ${4}"
echo "Neurons: ${5}"

sbatch -A pacer -t 10:00:00 -N 1 -p ${1} --gres=gpu:${2} -J GC_p${1}_ng${2}_nc${3}_vng${4}_n${5} -o ../../../logs/GraphChallenge/sbatch/out_p${1}_ng${2}_nc${3}_vng${4}_n${5}.txt -e ../../../logs/GraphChallenge/sbatch/err_p${1}_ng${2}_nc${3}_vng${4}_n${5}.txt run.sh ${1} ${2} ${3} ${4} ${5}