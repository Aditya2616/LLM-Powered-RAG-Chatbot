FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    RAG_DATA_DIR=/data \
    RAG_WEB_DIR=/app/docs \
    RAG_SAMPLE_DIR=/app/corpus/docs

WORKDIR /app

# CPU-only PyTorch first, so the image does not pull the multi-gigabyte CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[local]"

COPY docs ./docs
COPY corpus ./corpus

VOLUME ["/data", "/models"]
EXPOSE 8000

# Models are downloaded on first use into /models (mount a volume to keep them).
CMD ["rag-chatbot", "serve", "--host", "0.0.0.0", "--port", "8000"]
