from io import BytesIO
from PyPDF2 import PdfReader

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join((p.extract_text() or "") for p in reader.pages)
