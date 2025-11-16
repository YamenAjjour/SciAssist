#!/bin/bash
source activate base
conda init
#conda activate vllm
conda activate sciassist-new
fastapi run scirag/setup_api.py --port 8585
