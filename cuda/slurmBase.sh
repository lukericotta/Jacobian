#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-00:50:00
#SBATCH -o outBase
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

date

module load cuda cmake gcc/recommended
cmake .
make

rm -rf cudaBaseLog

for (( m=2048; m<=2048; m=m*2 ))
do
   for (( n=4096*8; n<=4096*8; n=n*2 ))
   do
      ../scripts/generateInput.py $n $m inputBase
      nvprof --log-file profBase ./jacobianBase inputBase outputBase 1
   done
done

