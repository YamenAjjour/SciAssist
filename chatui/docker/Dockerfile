FROM python:3.9-slim

WORKDIR /sciassist

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /sciassist

RUN pip3 install streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "/sciassist/chatui/setup_simple_chatui.py", "--server.port=8501", "--server.address=0.0.0.0"]