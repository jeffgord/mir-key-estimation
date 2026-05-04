#!/bin/bash

#SBATCH --job-name=key-estimation-krumhansl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_932_general
#SBATCH --output=logs/%j.out

singularity exec --fakeroot \
  --overlay $SCRATCH/key-estimation-scratch.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python krumhansl_predict_keys.py --data-dir $SCRATCH/fma_keys --out-dir $SCRATCH/key-estimation-scratch/out --workers 8"
