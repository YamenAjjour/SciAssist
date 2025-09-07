#!/bin/bash
source activate base 
#conda activate vllm
conda activate sciassist
fastapi run scirag/setup_api.py --port 8585
