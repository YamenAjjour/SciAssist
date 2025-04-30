#!/bin/bash -l
#SBATCH --job-name=fine-tuning-2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output setup_rag.out
#SBATCH --error setup_rag.err
#SBATCH --gpus=1
module load Miniforge3
conda activate sciassist

python scirag/setup_rag.py --debug --path-index data/index --path-dataset data/acl-publication-info.74k.parquet
  --path-model /home/yajjour/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B