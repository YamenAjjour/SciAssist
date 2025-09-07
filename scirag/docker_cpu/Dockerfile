FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during apt-get.
RUN apt-get update && apt-get install -y \
    python3-pip

RUN pip install langchain langchain-community faiss-gpu-cu12 fastapi[standard] pandas tqdm sentence-transformers vllm fastparquet langchain_huggingface langchain-google-genai python-dotenv
WORKDIR /sciassist
COPY . /sciassist
CMD ["fastapi", "run", "/sciassist/scirag/setup_api.py", "--port", "80"]
