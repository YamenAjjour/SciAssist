FROM sciassist-base-img
WORKDIR /sciassist
COPY . /sciassist
CMD ["fastapi", "run", "/sciassist/scirag/setup_api.py", "--port", "80"]
