FROM python:3.12-slim
WORKDIR /app
COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-lock.txt
COPY rag ./rag
COPY web ./web
COPY samples ./samples
COPY eval ./eval
COPY .env.example ./
ENV RAG_CONFIG_DIR=/data RAG_DATA_DIR=/data/knowledge
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "rag.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
