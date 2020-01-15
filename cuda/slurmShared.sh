#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-00:50:00
#SBATCH -o outShared
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

date

module load cuda cmake gcc/recommended
cmake .
make

rm -rf cudaSharedLog

for (( m=256; m<=2048; m=m*2 ))
do
   for (( n=4096; n<=4096*4; n=n*2 ))
   do
      ../scripts/generateInput.py $n $m inputShared
      ./jacobianShared inputShared outputShared 1
   done
done

