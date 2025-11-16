#!/bin/bash
source activate base
conda init
conda activate sciassist-new
streamlit run chatui/setup_simple_answer_form.py  --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none
