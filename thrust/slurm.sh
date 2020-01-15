#!/usr/bin/bash

#SBATCH -p slurm_shortgpu
#SBATCH -t 0-00:50:00
#SBATCH -o output
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

date

module load cuda cmake gcc/recommended
cmake .
make

rm -rf ThustLog

for (( m=10; m<=10; m=m*2 ))
do
   for (( n=1; n<=1; n=n*2 ))
   do
      ../scripts/generateInput.py $n $m in_thrust
      nvprof --log-file log_thrust ./jacobian2 in_thrust out_thrust 1
   done
done

