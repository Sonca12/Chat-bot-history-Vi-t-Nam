"""
Adaptive Chunking Module - Inspired by ekimetrics/adaptive-chunking
Tien xu ly va phan doan thong minh cho tai lieu Lich Su Viet Nam (1986-2000)
- Tu dong phat hien tieu de chuong, muc, tieu muc
- Giu nguyen ven cau va doan van, tranh ngat quang ngu nghia
- Gan metadata: so trang PDF, thuc the lich su, thoi gian, chu de
"""

import os
import re
import json
import unicodedata
from pathlib import Path

# Danh sach thuc the lich su quan trong can theo doi va gan the
THUC_THE_LICH_SU = {
    "nhan_vat": [
        "Trường Chinh", "Phạm Văn Đồng", "Nguyễn Văn Linh", "Võ Văn Kiệt",
        "Đỗ Mười", "Lê Đức Anh", "Phan Văn Khải", "Trần Đức Lương",
        "Lê Khả Phiêu", "Võ Nguyên Giáp", "Nguyễn Cơ Thạch", "Nguyễn Mạnh Cầm",
        "Vũ Khoan", "Hồ Chí Minh", "Lê Duẩn", "Nguyễn Ái Quốc", "Bill Clinton",
        "Giang Trạch Dân", "Lý Bằng", "Chu Dung Cơ", "Hun Sen", "Lý Quang Diệu"
    ],
    "su_kien": [
        "Đổi mới", "Đại hội VI", "Đại hội VII", "Đại hội VIII", "Đại hội IX",
        "Khoán 10", "Bình thường hóa quan hệ", "Gia nhập ASEAN", "APEC",
        "Hiệp định Thương mại Việt - Mỹ", "Hiệp ước Biên giới trên đất liền",
        "Phân định Vịnh Bắc Bộ", "Đường dây 500kV", "ASEM", "Cương lĩnh 1991"
    ],
    "to_chuc_quoc_gia": [
        "ASEAN", "APEC", "WTO", "Hoa Kỳ", "Mỹ", "Trung Quốc", "Liên Xô",
        "Liên bang Nga", "Nhật Bản", "EU", "Liên minh Châu Âu", "Campuchia",
        "Lào", "Cuba", "Pháp", "Đức"
    ]
}


def chuan_hoa_chu(text: str) -> str:
    """Loai bo dau va dua ve chu thuong de so khop thuc the."""
    t = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn").replace("đ", "d")


def trich_xuat_thuc_the(doan_van: str) -> dict:
    """Phat hien cac thuc the lich su xuat hien trong doan van."""
    text_lower = doan_van.lower()
    text_clean = chuan_hoa_chu(doan_van)
    
    ket_qua = {
        "nhan_vat": [],
        "su_kien": [],
        "nam_thoi_gian": []
    }
    
    for nv in THUC_THE_LICH_SU["nhan_vat"]:
        nv_clean = chuan_hoa_chu(nv)
        if nv_clean in text_clean:
            ket_qua["nhan_vat"].append(nv)
            
    for sk in THUC_THE_LICH_SU["su_kien"]:
        sk_clean = chuan_hoa_chu(sk)
        if sk_clean in text_clean:
            ket_qua["su_kien"].append(sk)
            
    # Tim cac nam tu 1900 - 2025
    nam_tim_thay = re.findall(r'\b(19\d{2}|20\d{2})\b', doan_van)
    if nam_tim_thay:
        ket_qua["nam_thoi_gian"] = sorted(list(set(nam_tim_thay)))
        
    return ket_qua


def cat_cau(text: str) -> list[str]:
    """Tach cau tieng Viet chuan xac khong ngat o cac tu viet tat."""
    text = re.sub(r'(PGS|TS|NCVCC|NCVC|ThS|GS|Sđd|tr)\.', r'\1<DOT>', text)
    cau_list = re.split(r'([.?!;]\s+|\n\n+)', text)
    
    cau_hoan_chinh = []
    tam = ""
    for c in cau_list:
        tam += c
        if re.search(r'[.?!;]\s+$|\n\n+$', c) or len(c) > 300:
            tam_khoi_phuc = tam.replace('<DOT>', '.')
            if tam_khoi_phuc.strip():
                cau_hoan_chinh.append(tam_khoi_phuc.strip())
            tam = ""
    if tam.strip():
        cau_hoan_chinh.append(tam.replace('<DOT>', '.').strip())
        
    return [c for c in cau_hoan_chinh if len(c) > 5]


def thuc_hien_adaptive_chunking(
    input_json="data/extracted_text/thong_tin.json",
    output_json="data/chunks/chunks_adaptive.json",
    target_size=850,
    min_size=350,
    max_size=1350,
    overlap_sentences=1
) -> list[dict]:
    """
    Thuat toan Adaptive Chunking:
    - Di chuyen theo cau truc trang va doan van tu thong_tin.json
    - Tu dong nhan dien tieu de va gop/chia linh hoat dua tren ngu nghia
    - Dam bao cac cau khong bi cat ngang
    """
    path_in = Path(input_json)
    if not path_in.exists():
        raise FileNotFoundError(f"Khong tim thay {input_json}")
        
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pages = data.get("chi_tiet", [])
    chunks = []
    chunk_idx = 0
    
    current_title = "Lịch sử Việt Nam (1986 - 2000)"
    current_chapter = "Chung"
    
    # Nhan dien tieu de va chuong
    re_chapter = re.compile(r'(CHƯƠNG\s+[IVXLCDM\d]+[^\n]*|Chương\s+[IVXLCDM\d]+[^\n]*)', re.IGNORECASE)
    re_section = re.compile(r'(\n\d+\.\s+[^\n]{4,80}|\n[A-ZIVX]+\.\s+[^\n]{4,80})')
    
    buffer_sentences = []
    buffer_pages = []
    
    for page_item in pages:
        trang = page_item.get("trang", 1)
        raw_text = page_item.get("text", "").strip()
        if not raw_text or len(raw_text) < 40:
            continue
            
        # Phat hien Chuong moi
        m_chap = re_chapter.search(raw_text)
        if m_chap:
            current_chapter = m_chap.group(1).strip()
            current_title = current_chapter
            
        # Phat hien Muc moi
        m_sec = re_section.search(raw_text)
        if m_sec:
            current_title = f"{current_chapter} - {m_sec.group(1).strip()}"
            
        sentences = cat_cau(raw_text)
        for s in sentences:
            buffer_sentences.append(s)
            buffer_pages.append(trang)
            
            # Kiem tra do dai hien tai trong buffer
            tong_ky_tu = sum(len(x) for x in buffer_sentences)
            if tong_ky_tu >= target_size:
                # Tao 1 chunk moi
                text_chunk = " ".join(buffer_sentences)
                entities = trich_xuat_thuc_the(text_chunk)
                start_p = buffer_pages[0]
                end_p = buffer_pages[-1]
                
                chunk_obj = {
                    "id": f"adp_{chunk_idx}",
                    "loai": "adaptive",
                    "tieu_de": current_title,
                    "chuong": current_chapter,
                    "trang_dau": start_p,
                    "trang_cuoi": end_p,
                    "text": text_chunk,
                    "so_ky_tu": len(text_chunk),
                    "thuc_the": entities
                }
                chunks.append(chunk_obj)
                chunk_idx += 1
                
                # Giu lai overlap_sentences cho chunk tiep theo de bao toan ngu canh
                if overlap_sentences > 0 and len(buffer_sentences) > overlap_sentences:
                    buffer_sentences = buffer_sentences[-overlap_sentences:]
                    buffer_pages = buffer_pages[-overlap_sentences:]
                else:
                    buffer_sentences = []
                    buffer_pages = []

    # Xử lý phần còn lại trong buffer
    if buffer_sentences:
        text_chunk = " ".join(buffer_sentences)
        if len(text_chunk) > 100:
            entities = trich_xuat_thuc_the(text_chunk)
            chunk_obj = {
                "id": f"adp_{chunk_idx}",
                "loai": "adaptive",
                "tieu_de": current_title,
                "chuong": current_chapter,
                "trang_dau": buffer_pages[0] if buffer_pages else 1,
                "trang_cuoi": buffer_pages[-1] if buffer_pages else 1,
                "text": text_chunk,
                "so_ky_tu": len(text_chunk),
                "thuc_the": entities
            }
            chunks.append(chunk_obj)
            
    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"[*] Adaptive Chunking hoan tat: Tao thanh cong {len(chunks)} chunks!")
    return chunks


if __name__ == "__main__":
    thuc_hien_adaptive_chunking()
