FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#RUN apt-get update && apt-get install -y \
#    python3-pip
#RUN pip install langchain langchain-community vllm
RUN pip install langchain langchain-community faiss-gpu-cu12 fastapi[standard] pandas tqdm sentence-transformers vllm
WORKDIR /sciassist
COPY . /sciassist
CMD ["fastapi", "run", "/sciassist/scirag/setup_api.py", "--port", "80"]
