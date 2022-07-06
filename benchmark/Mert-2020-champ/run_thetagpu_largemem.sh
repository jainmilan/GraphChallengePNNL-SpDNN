export DATASET=/lus/grand/projects/GRACE/spdnn/dataset
export INPUT=392191985
export BIAS=-0.45
export NEURON=65536
export LAYER=1920
export BATCH=60000

export BLOCKSIZE=256
export BUFFER=24

export OMP_NUM_THREADS=16


for layer in 120 480 1920 
do 
	for neuron in 1024 4096 16384 65536 
	do 
		if [[ $neuron -eq 1024 ]]
		then 
			export BIAS=-0.3
			export INPUT=6374505
		fi
		if [[ $neuron -eq 4096 ]]
		then 
			export BIAS=-0.35
			export INPUT=25019051
		fi
		if [[ $neuron -eq 16384 ]]
		then 
			export BIAS=-0.4
			export INPUT=98858913
		fi
		if [[ $neuron -eq 65536 ]]
		then 
			export BIAS=-0.45
			export INPUT=392191985
		fi

		export NEURON=$neuron
		export LAYER=$layer

		echo $LAYER
		echo $NEURON
		echo $BIAS
		echo $INPUT
		echo $DATASET
		./inference &> ${LAYER}_${NEURON}_output.log

		echo "****************************************************"

	done 

done

