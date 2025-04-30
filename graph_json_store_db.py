import json
import chromadb
import uuid
import os

# JSON 파일 경로
GRAPH_JSON_PATH = "subject_graph.json"  # 같은 디렉토리에 있는 파일

# ChromaDB 경로
CHROMA_PATH = "./chroma_db"

# 컬렉션 이름
COLLECTION_NAME = "education_graph"

# JSON 파일 로드
with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

# ChromaDB 초기화 및 컬렉션 초기화
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# 기존 컬렉션이 있으면 삭제
if COLLECTION_NAME in chroma_client.list_collections():
    chroma_client.delete_collection(name=COLLECTION_NAME)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# 1. 노드 정보 저장
for node in graph_data["nodes"]:
    doc_text = f"과목명: {node['id']}, 우선순위: {node['priority']}"
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc_text],
        metadatas=[{"type": "node", "course": node["id"], "priority": node["priority"]}]
    )

# 2. 간선 정보 저장
for edge in graph_data["edges"]:
    doc_text = f"'{edge['from']}' 과목은 '{edge['to']}' 과목의 선수과목입니다. 트랙: {edge['track']}"
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc_text],
        metadatas=[{"type": "edge", "track": edge["track"], "from": edge["from"], "to": edge["to"]}]
    )

# 3. 위상 정렬 저장
order_text = "과목을 듣는 추천 순서: " + " → ".join(graph_data["topological_order"])
collection.add(
    ids=[str(uuid.uuid4())],
    documents=[order_text],
    metadatas=[{"type": "topological_order"}]
)

# 4. 최단 경로들 저장
for key, path in graph_data["shortest_paths"].items():
    doc_text = f"{key} 최단경로: {' → '.join(path)}"
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc_text],
        metadatas=[{"type": "shortest_path", "from_to": key}]
    )

print(" subject_graph.json 데이터가 ChromaDB에 저장되었습니다!")
