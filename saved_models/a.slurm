#!/bin/bash
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=li.yang.pbs@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

echo 'My job ran on:'
echo $SLURM_NODELIST
echo 'submission dir:' $SLURM_SUBMIT_DIR
echo modules:
module list
cd $SLURM_SUBMIT_DIR

export PYTHONPATH=../:../mask_rcnn:../ensemble

cd ../mask_rcnn
./build_lib.sh
cd -

srun python predict.py
