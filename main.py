import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb

# .env 파일 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#  Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")

#  ChromaDB 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#  두 개의 컬렉션을 불러옴
collection_report = chroma_client.get_or_create_collection(name="education_report")
collection_graph = chroma_client.get_or_create_collection(name="education_graph")

def get_gemini_answer(query, top_k=10):
    """ChromaDB에서 두 컬렉션을 검색 후 Gemini로 자연스러운 답변 생성"""
    
    #  두 컬렉션에서 각각 유사 문서 검색
    results_report = collection_report.query(query_texts=[query], n_results=top_k)
    results_graph = collection_graph.query(query_texts=[query], n_results=top_k)

    #  문서 통합
    retrieved_docs = []
    if results_report["documents"]:
        retrieved_docs.extend(results_report["documents"][0])
    if results_graph["documents"]:
        retrieved_docs.extend(results_graph["documents"][0])
    if not retrieved_docs:
        retrieved_docs = ["관련된 정보를 찾을 수 없습니다."]

    context = "\n".join(retrieved_docs)

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
    내용 강조나 리스트 표현이 필요하다면 부드러운 문장으로 자연스럽게 표현해 주세요.  
    예: "가장 중요한 과목은 ~입니다." 또는 "추천 순서는 ~입니다." 와 같이 문장으로 표현해 주세요.

    가독성이 좋도록 줄바꿈을 적절히 활용하고, 말이 너무 길지 않도록 문단을 나눠주세요.
    아이콘을 활용해서 가독성을 향상시키는것도 좋은 방법이에요.

    다음은 참고 정보입니다:
    {context}

    학생의 질문:
    {query}
    """

    response = model.generate_content(prompt)
    return response.text

#  사용자 입력 기반 실행
if __name__ == "__main__":
    print("안녕하세요! 컴공도우미봇 입니다. 어떤 질문을 하고싶으신가요?")
    query_text = input("\n질문: ")
    answer = get_gemini_answer(query_text)
    print("\n🔹 컴공도우미봇 답변:")
    # print(answer.replace('**', ''))
