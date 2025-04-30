import pdfplumber
import chromadb
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

_path = os.path.dirname(os.path.abspath(__file__))


# PDF → 텍스트 변환
def extract_text_from_pdf(pdf_path: str):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text


# 텍스트 → 파일로 저장
def save_text_to_file(text: str, filename: str = "pdf텍스트변환본.txt"):
    output_path = os.path.join(_path, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f" 변환된 텍스트가 '{filename}'로 저장되었습니다!")


# 청킹(Chunking) 수행
def chunk_text(text: str, chunk_size=700, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    return chunks


# ChromaDB 저장 (기존 컬렉션 삭제 후 재생성)
def store_in_chromadb(chunks):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # 기존 collection 이름 문자열 리스트로 확인
    existing = chroma_client.list_collections()
    if "education_report" in existing:
        print("기존 CromaDB 컬렉션 삭제를 진행합니다.")
        chroma_client.delete_collection(name="education_report")

    # 새로 생성
    collection = chroma_client.get_or_create_collection(name="education_report")

    for idx, chunk in enumerate(chunks):
        unique_id = str(uuid.uuid4())
        collection.add(
            ids=[unique_id],
            documents=[chunk],
            metadatas=[{"source": "역량중심 교육과정 보고서", "chunk_index": idx}]
        )
    print(" PDF 데이터가 ChromaDB에 초기화 후 저장되었습니다!")



# 실행: PDF → 텍스트 변환 → 저장 → 청킹 → ChromaDB 저장
pdf_path = os.path.join(_path, "역량중심 교육과정 개발 보고서_컴퓨터공학과_축약.pdf")
pdf_text = extract_text_from_pdf(pdf_path)                     # 1️ PDF -> 텍스트 변환
# save_text_to_file(pdf_text, "pdf텍스트변환본.txt")            # 2️ 텍스트 파일 저장 (선택)
chunks = chunk_text(pdf_text, chunk_size=700, overlap=100)     # 3️ 청킹 수행
store_in_chromadb(chunks)                                      # 4️ ChromaDB에 저장
