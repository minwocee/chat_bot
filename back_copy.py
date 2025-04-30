from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# ─── 환경변수 로드 ──────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Gemini API 설정 ────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")    # gemini-1.5-pro    gemini-2.0-pro-exp-02-05

# ─── ChromaDB 클라이언트 및 컬렉션 초기화 ────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection_semantic = chroma_client.get_or_create_collection(
    name="semantic_education_chunks",
    embedding_function=embedding_fn
)

# ─── FastAPI 앱 생성 ────────────────────────────────────
app = FastAPI()

# ─── CORS 설정 (React 프론트 허용) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 요청 바디 모델 정의 ────────────────────────────────
class QueryRequest(BaseModel):
    query: str

# ─── 질문 처리 엔드포인트 ───────────────────────────────
@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query.strip()
    top_k = 10

    try:
        results_semantic = collection_semantic.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        return {"answer": f"DB 검색 중 오류가 발생했습니다: {str(e)}"}

    retrieved_docs = []
    if results_semantic.get("documents"):
        retrieved_docs.extend(results_semantic["documents"][0])

    if not retrieved_docs:
        retrieved_docs = ["관련된 정보를 찾을 수 없습니다."]

    # 중복 제거 및 길이 제한
    retrieved_docs = list(dict.fromkeys(retrieved_docs))[:3]
    context = "\n".join(retrieved_docs)[:1500]

    prompt = f"""
    당신은 컴퓨터공학과 신입생들의 과목 선택을 도와주는 조교 챗봇입니다.

    학생들은 과목 순서, 트랙 구성, 진로와 관련된 수업을 고민하고 있으며,  
    당신은 이들에게 마치 상담 선생님처럼 부드럽고 명확하게 설명해 줍니다.

    기술 용어(예: 위상정렬, 최단경로)는 사용하지 말고,  
    자연스럽고 따뜻한 말투로 설명해주세요.  
    "먼저 ~을 듣고, 그 다음 ~을 듣는 게 좋아요" 같은 말투를 사용해 주세요.  
    중요한 과목은 강조해도 좋아요.  
    학생들이 헷갈리지 않도록 순서를 정리해서 말해 주세요.

    주의: 답변에는 `*`, `**`, `-`, `•` 등과 같은 특수 기호를 사용하지 말아 주세요.  
    필요한 경우 부드러운 문장으로 자연스럽게 강조해주세요.

    다음은 참고 정보입니다:
    {context}

    학생의 질문:
    {query}
    """

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Gemini 응답 생성 중 오류가 발생했습니다: {str(e)}"}

# ─── 기본 루트 테스트 ──────────────────────────────────
@app.get("/")
def root():
    return {"message": "컴공도우미봇 API가 실행 중입니다 🚀"}