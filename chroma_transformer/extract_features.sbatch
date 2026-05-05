#!/bin/bash

#SBATCH --job-name=extract-chroma-features
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=logs/%j.out

singularity exec --fakeroot \
  --overlay $SCRATCH/key-estimation-scratch.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python extract_features.py --data-dir $SCRATCH/fma_keys --output-dir $SCRATCH/key-estimation-scratch/features --n-jobs 16"
