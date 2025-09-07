#FROM python:3.12-slim-bookworm
#COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during apt-get.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
#RUN apt-get update && apt-get install -y \
#    python3-pip
#RUN pip install langchain langchain-community vllm
# RUN pip install langchain langchain-community faiss-gpu-cu12 fastapi[standard] pandas tqdm sentence-transformers pyarrow fastparquet langchain_huggingface
# RUN apt-get update  -y
# RUN apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof
# RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
#
# RUN git clone https://github.com/vllm-project/vllm.git /tmp/vllm_source
# RUN cd /tmp/vllm_source
# RUN uv venv --python 3.12 --seed
#
# RUN uv pip install -r /tmp/vllm_source/requirements/cpu-build.txt --torch-backend auto --index-strategy unsafe-best-match
# RUn uv pip install -r /tmp/vllm_source/requirements/cpu.txt --torch-backend auto
RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install vllm[cpu]
WORKDIR /sciassist
COPY . /sciassist
#CMD ["python", "print(10)"]
#CMD ["fastapi", "run", "/sciassist/scirag/setup_api.py", "--port", "80"]
