#!/usr/bin/env bash

#SBATCH --job-name=LukesJob

#SBATCH --output=/srv/home/lukericotta/me759-ricotta/jobs/job_output-%j.txt

#SBATCH --sockets-per-node=1 -t 0-00:00:30 -p slurm_shortgpu

#SBATCH --cpus-per-task=16

#SBATCH --cores-per-socket=1

#SBATCH --threads-per-core=1

cd $SLURM_SUBMIT_DIR

./jacobian in out .0001
