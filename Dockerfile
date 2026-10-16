FROM python:3.11-slim

WORKDIR /app

# Cai dat cac thu vien can thiet cho Offline RAG Engine
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-multipart python-dotenv

# Sao chep toan bo ma nguon va du lieu
COPY . .

# Mo cong mac dinh (8000 cho Render/cuc bo, 7860 cho Hugging Face Spaces)
EXPOSE 8000
EXPOSE 7860

# Khoi chay ung dung
CMD ["python", "run.py"]
