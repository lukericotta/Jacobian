#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-05:00:00
#SBATCH -o outShared2
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

date

module load cuda cmake gcc/recommended
cmake .
make

rm -rf cudaShared2Log

for (( m=256; m<=4096; m=m*2 ))
do
   for (( n=256; n<=4096*8; n=n*2 ))
   do
      ../scripts/generateInput.py $n $m inputShared2
      ./jacobianShared2 inputShared2 outputShared2 1
   done
done

