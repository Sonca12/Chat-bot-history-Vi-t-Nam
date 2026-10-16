"""
Entrypoint khoi chay ung dung RAG Chatbot Lich Su Viet Nam
Ho tro tu dong nhan PORT tren cac nen tang Cloud (Render, Hugging Face, Railway, Heroku...)
Chay cuc bo: python run.py
"""

import os
import sys
from pathlib import Path

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Dam bao thu muc goc va thu muc code duoc them vao sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "code"))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"========================================================")
    print(f"  RAG Lịch Sử Việt Nam (1986 - 2000) - 100% Offline RAG ")
    print(f"  Server dang khoi chay tai: http://localhost:{port}")
    print(f"========================================================")
    uvicorn.run("code.api:app", host="0.0.0.0", port=port, reload=False)
