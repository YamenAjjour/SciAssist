FROM openeuler/vllm-cpu

#RUN apt-get update && apt-get install -y \
#    python3-pip
#RUN pip install langchain langchain-community vllm
RUN pip install langchain langchain-community faiss-gpu-cu12 fastapi[standard] pandas tqdm sentence-transformers pyarrow fastparquet langchain_huggingface
WORKDIR /sciassist
COPY . /sciassist
CMD ["fastapi", "run", "/sciassist/scirag/setup_api.py", "--port", "80"]
