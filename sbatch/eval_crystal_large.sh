#!/bin/bash
#SBATCH --job-name=eval_crystal_large
#SBATCH --partition=devlab
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/crystal/logs/%J.%x.out"

wrapper="sbatch/eval_rainier-v3.9.sh.wrapper"
cat $0
echo "--------------------"
cat $wrapper
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label ${wrapper} \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    "../runs/[RUN_NAME]/model/ckp_[BEST_CKPT].pth"
