#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-03:00:00
#SBATCH -o out
#SBATCH -e err
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load pgi/18.10

for (( m=256; m<=8192; m=m*2 ))
do
    for (( n=256; n<=4096*8; n=n*2 ))
    do
	../scripts/generateInput.py $n $m input

	$CXX -fast -O3 -ta=tesla:managed,ccall -Minfo=accel -std=c++17 -lm jacobian.cpp -o jacobian

	./jacobian input output 1
    done
done
