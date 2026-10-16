"""
FastAPI — RAG Chatbot Lich Su Viet Nam (100% Offline RAG + Hybrid BM25 + Knowledge Graph)
Hoat dong doc lap, khong can API ngoai, khong lo het han API.
Chay tu thu muc goc: python -m uvicorn --app-dir code api:app --reload --port 8000
"""

import os
import json
import tempfile
import traceback
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import lay_engine_rag
from knowledge_graph import KNOWLEDGE_GRAPH

app = FastAPI(
    title="RAG Lịch Sử Việt Nam (1986 - 2000)",
    description="Hệ thống RAG hỏi đáp lịch sử Việt Nam 100% Offline với Adaptive Chunking & Knowledge Graph",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khoi tao RAG engine ngay khi start
engine = lay_engine_rag()


# ============================================================
#  Schemas
# ============================================================
class YeuCau(BaseModel):
    question: str
    top_k: int = 5


class PhanHoi(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    route: str | None = "rag_offline"
    suggestions: list[str] | None = None
    entity: str | None = None


# ============================================================
#  Endpoints
# ============================================================
@app.get("/")
def trang_chu():
    ui_path = Path(__file__).resolve().parent.parent / "chat_ui.html"
    return FileResponse(str(ui_path))


@app.get("/health")
def kiem_tra():
    return {
        "status": "ok",
        "service": "RAG Lịch Sử Việt Nam (1986 - 2000)",
        "mode": "100% Offline (Không phụ thuộc API ngoài)",
        "modules": {
            "adaptive_chunks_loaded": len(engine.chunks),
            "qa_pairs_loaded": len(engine.qa_pairs),
            "knowledge_graph_entities": len(KNOWLEDGE_GRAPH),
            "bm25_retriever": engine.bm25 is not None,
            "offline_ready": True
        },
    }


@app.post("/ask", response_model=PhanHoi)
def hoi(req: YeuCau):
    cau_hoi = req.question.strip()
    if not cau_hoi:
        raise HTTPException(400, "Câu hỏi không được để trống")
    try:
        kq = engine.tra_loi(cau_hoi, top_k=req.top_k)
        return PhanHoi(
            question=cau_hoi,
            answer=kq["answer"],
            sources=kq.get("sources", []),
            route=kq.get("route", "rag_offline"),
            suggestions=kq.get("suggestions", []),
            entity=kq.get("entity")
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi xử lý câu hỏi: {str(e)}")


@app.post("/ask/image", response_model=PhanHoi)
async def hoi_bang_anh(
    text: str = Form(default=""),
    image: UploadFile = File(default=None),
    top_k: int = Form(default=5),
):
    cau_hoi = ""
    tmp = []

    try:
        if image and image.filename:
            suffix = Path(image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await image.read())
                tmp.append(f.name)

            try:
                import pytesseract
                from PIL import Image as PILImage
                img = PILImage.open(tmp[0])
                cau_hoi = pytesseract.image_to_string(img, lang="vie").strip()
            except Exception as ocr_err:
                print(f"[!] OCR khong kha dung: {ocr_err}")
                cau_hoi = text.strip() if text else "Trương Trinh làm gì vào giai đoạn đó?"

        elif text:
            cau_hoi = text.strip()

        if not cau_hoi:
            raise HTTPException(400, "Cần cung cấp câu hỏi bằng văn bản hoặc hình ảnh")

        kq = engine.tra_loi(cau_hoi, top_k=top_k)
        return PhanHoi(
            question=cau_hoi,
            answer=kq["answer"],
            sources=kq.get("sources", []),
            route=kq.get("route", "rag_offline"),
            suggestions=kq.get("suggestions", []),
            entity=kq.get("entity")
        )

    finally:
        for p in tmp:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/chunks")
def danh_sach_chunks(limit: int = 50, offset: int = 0, query: str = ""):
    """Endpoint tien ich: Cho phep UI kham pha va tra cuu cac Adaptive Chunks."""
    ds = engine.chunks
    if query.strip():
        q_clean = unicodedata.normalize("NFD", query.lower())
        q_clean = "".join(c for c in q_clean if unicodedata.category(c) != "Mn")
        ds = [
            c for c in ds
            if q_clean in unicodedata.normalize("NFD", c.get("text", "").lower())
            or q_clean in unicodedata.normalize("NFD", c.get("tieu_de", "").lower())
        ]
    
    total = len(ds)
    cat_doan = ds[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "chunks": [
            {
                "id": c.get("id"),
                "tieu_de": c.get("tieu_de"),
                "chuong": c.get("chuong"),
                "trang_dau": c.get("trang_dau"),
                "trang_cuoi": c.get("trang_cuoi"),
                "so_ky_tu": c.get("so_ky_tu"),
                "tom_tat": c.get("text", "")[:250] + "...",
                "thuc_the": c.get("thuc_the", {})
            }
            for c in cat_doan
        ]
    }


@app.get("/entities")
def danh_sach_thuc_the():
    """Endpoint tien ich: Tra ve so do thuc the va moi quan he mat thiet."""
    return {
        "total": len(KNOWLEDGE_GRAPH),
        "entities": [
            {
                "id": k,
                "ten": v["ten"],
                "chuc_vu": v["chuc_vu"],
                "thoi_ky": v["giai_doan_trong_tai_lieu"],
                "quan_he_mat_thiet": [qh["nhan_vat"] for qh in v.get("quan_he_mat_thiet", [])]
            }
            for k, v in KNOWLEDGE_GRAPH.items()
        ]
    }


# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
