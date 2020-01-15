#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-01:00:00
#SBATCH -o outTest
#SBATCH -e errTest
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load pgi/18.10 

# export PGI_ACC_TIME=1
# export PGI_ACC_NOTFY=2

$CXX -fast -O3 -ta=tesla:managed,ccall -Minfo=accel -std=c++17 -lm jacobian.cpp -o jacobian

../scripts/generateInput.py 16384 4096 inputTest
./jacobian inputTest outputTest 1
