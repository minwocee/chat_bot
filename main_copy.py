import os
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ─── 환경변수 로드 ──────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Gemini API 설정 ─────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")

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

TOP_K = 5  # 상위 5개 검색

def get_gemini_answer(query, top_k=TOP_K):
    query = query.strip()
    if not query:
        return "질문을 입력해 주세요."

    try:
        # semantic_education_chunks 컬렉션에서만 검색
        results_semantic = collection_semantic.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        return f"DB 검색 중 오류가 발생했습니다: {str(e)}"

    retrieved_docs = []
    if results_semantic.get("documents"):
        retrieved_docs.extend(results_semantic["documents"][0])

    if not retrieved_docs:
        retrieved_docs = ["관련된 정보를 찾을 수 없습니다."]

    # 중복 제거 및 길이 제한
    retrieved_docs = list(dict.fromkeys(retrieved_docs))[:3]
    context = "\n".join(retrieved_docs)[:1500]  # 최대 1500자

    # 간단한 프롬프트 구성
    prompt = f"""
    참고 정보:
    {context}

    학생 질문:
    {query}

    학생에게 친절하고 명확하게 답변해 주세요.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 답변 생성 중 오류가 발생했습니다: {str(e)}"

# ─── 사용자 입력 기반 실행 ───────────────────────────────
if __name__ == "__main__":
    print("안녕하세요! 컴공도우미봇 입니다. 어떤 질문을 하고싶으신가요?")
    query_text = input("\n질문: ")
    answer = get_gemini_answer(query_text)
    print("\n[컴공도우미봇 답변]:")
    print(answer)
