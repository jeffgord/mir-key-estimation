#!/bin/bash

#SBATCH --job-name=key-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=logs/train_%a.out
#SBATCH --array=0-9

singularity exec --fakeroot --nv \
  --overlay $SCRATCH/key-estimation-scratch/conda-overlay.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; cd $SCRATCH/key-estimation-scratch; python chroma_transformer.py $SLURM_ARRAY_TASK_ID"
