#!/bin/bash
source activate base 
conda activate vllm
fastapi run scirag/setup_api.py --port 8585
