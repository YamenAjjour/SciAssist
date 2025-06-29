#!/bin/bash -l
#SBATCH --job-name=setup_rag_indx
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output setup_rag.out
#SBATCH --error setup_rag.err
#SBATCH --gpus=2
module load Miniforge3
conda activate sciassist

python scirag/setup_rag.py --path-index /mnt/home/yajjour/SciAssist/data/index --path-dataset /mnt/home/yajjour/SciAssist/data/acl-publication-info.74k.parquet --path-model /mnt/home/yajjour/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B