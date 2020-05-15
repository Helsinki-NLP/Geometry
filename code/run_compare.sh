#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --mem=350000
#SBATCH --mem=350000
#SBATCH --gres=gpu:v100:1
#SBATCH -J bert
#SBATCH -o logs/bert.out.%j
#SBATCH -e logs/bert.err.%j
#SBATCH --account=project_2001194
#SBATCH

export TMPDIR=/scratch/project_2002233
export TEMP=/scratch/project_2002233
export TMP=/scratch/project_2002233

python compare_contextEmbeddings.py --cuda
