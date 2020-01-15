#!/usr/bin/env bash

#SBATCH -p slurm_shortgpu
#SBATCH --time=0-05:00:00
#SBATCH --output=outSeq
#SBATCH --ntasks=1 --cpus-per-task=8

date

cd $SLURM_SUBMIT_DIR

module load gcc cmake

cmake .
make

rm -rf sequentialLog

for (( m=256; m<=4096; m=m*2 ))
do
   for (( n=256; n<=4096*8; n=n*2 ))
   do
      ../scripts/generateInput.py $n $m inputSeq
      ./jacobian inputSeq outputSeq 1
   done
done
