FROM langchain/langchain
WORKDIR /sciassist
COPY . /sciassist
RUN pip install faiss-cpu fastapi[standard] pandas tqdm vllm
CMD ["fastapi", "run", "/sciassist/scirag/setup_api.py", "--port", "80"]
