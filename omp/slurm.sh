#!/usr/bin/env bash

#SBATCH -p slurm_shortgpu
#SBATCH --time=0-05:59:59
#SBATCH --output=outOmp
#SBATCH --ntasks=1 --cpus-per-task=16

date

cd $SLURM_SUBMIT_DIR

module load gcc cmake

cmake .
make

../scripts/generateInput.py $n $m inputOmp
./jacobian inputOmp outputOmp 1
