#!/usr/bin/env bash

#SBATCH --job-name=LukesJob

#SBATCH --output=p3_out.txt

#SBATCH --sockets-per-node=1 -t 0-00:10:00 -p slurm_shortgpu

#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:1

#SBATCH --cores-per-socket=1

#SBATCH --threads-per-core=1

cd $SLURM_SUBMIT_DIR

./problem3 1000
