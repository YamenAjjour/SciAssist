#!/bin/bash -l
#SBATCH --job-name=paired-task-learning
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output setup_rag.out
#SBATCH --error setup_rag.err

module load Miniforge3
conda activate sciassist

python scirag/setup_rag.py --path-index /mnt/home/yajjour/SciAssist/data/index --path-dataset /mnt/home/yajjour/SciAssist/data/acl-publication-info.74k.parquet --path-model /mnt/home/yajjour/pre-trained-models/TinyLlama-1.1B-Chat-v1.0 --query "cross domain robustness of argument mining "