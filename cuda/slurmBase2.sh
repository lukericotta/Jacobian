#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-05:00:00
#SBATCH -o outBase2
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

date

module load cuda cmake gcc/recommended
cmake .
make

rm -rf cudaBase2Log

for (( m=256; m<=8192; m=m*2 ))
do
   for (( n=256; n<=8192; n=n*2 ))
   do
      ../scripts/generateInput.py $n $m inputBase2
      ./jacobianBase2 inputBase2 outputBase2 1
   done
done

