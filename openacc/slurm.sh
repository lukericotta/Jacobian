#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-00:2:00
#SBATCH -o out
#SBATCH -e err
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load pgi/18.10 

# export PGI_ACC_TIME=1
# export PGI_ACC_NOTFY=2

$CXX -fast -O3 -ta=tesla:managed,ccall -Minfo=accel -std=c++17 -lm jacobian.cpp -o jacobian

./jacobian input output 1
