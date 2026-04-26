import time
import os
import warnings
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 這裡更新為新版套件

warnings.filterwarnings("ignore")

# ==========================================
# 1. 載入我們在 Part 1 建立的 Vector DB
# ==========================================
EMBED_MODEL_NAME = "BAAI/bge-m3"
print(f"Loading Embedding Model: {EMBED_MODEL_NAME} on CUDA...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Loading ChromaDB Index B from disk...")
vector_db = Chroma(
    persist_directory="./chroma_db_B",
    embedding_function=embeddings,
    collection_name="strategy_B"
)

# ==========================================
# 2. 載入 Cross-Encoder (Re-ranker)
# ==========================================
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading Cross-Encoder: {RERANK_MODEL_NAME} on CUDA...")
reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')

# ==========================================
# 3. 兩階段檢索函數 (加入計時與報告用 Log)
# ==========================================
def advanced_rag_retrieve(query, db, top_k_retrieval=20, top_k_rerank=3):
    print(f"\n" + "="*50)
    print(f"User Query: '{query}'")
    print("="*50)

    # --- Stage 1: Dense Retrieval (Vector Search) ---
    t0 = time.time()
    initial_docs = db.similarity_search(query, k=top_k_retrieval)
    t1 = time.time()
    vec_search_time = t1 - t0

    print(f"\n[Stage 1] Vector Search (Time taken: {vec_search_time:.4f} seconds)")
    print("--- Top 5 Original Vector Search Order ---")
    for i, doc in enumerate(initial_docs[:5]):
        # 把換行符號替換掉比較好閱讀
        clean_text = doc.page_content.replace("\n", " ")
        print(f"  [Rank {i+1}] {clean_text[:500]}...")

    # --- Stage 2: Cross-Encoder Re-ranking ---
    t2 = time.time()
    doc_texts = [d.page_content for d in initial_docs]
    pairs = [[query, doc_text] for doc_text in doc_texts]
    
    # 預測分數 (分數越高代表與 Query 越相關)
    scores = reranker.predict(pairs)
    
    # 將文件與分數綁定並排序
    scored_docs = list(zip(initial_docs, scores))
    scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    final_docs = [doc for doc, score in scored_docs_sorted[:top_k_rerank]]
    t3 = time.time()
    rerank_time = t3 - t2

    print(f"\n[Stage 2] Re-ranking (Time taken: {rerank_time:.4f} seconds)")
    print("--- Final Top 3 Re-ranked Results ---")
    for i, (doc, score) in enumerate(scored_docs_sorted[:top_k_rerank]):
        original_idx = initial_docs.index(doc)
        clean_text = doc.page_content.replace("\n", " ")
        print(f"  [New Rank {i+1} | Orig Rank {original_idx+1}] Score: {score:.4f} | Text: {clean_text[:500]}...") # 這裡改成 200
        
    return final_docs

# ==========================================
# 4. 測試執行 (換成 Kaggle 科學題庫的內容)
# ==========================================

# 測試 1：關於 MOND 理論 (對應 train.csv 的第 0 題)
# 我們故意用稍微換句話說的方式提問
test_query_1 = "How does Modified Newtonian Dynamics (MOND) affect the discrepancy of missing baryonic mass?"
print("\nTesting Query 1...")
advanced_rag_retrieve(test_query_1, vector_db)

# 測試 2：關於 Triskeles 符號 (對應 train.csv 的第 2 題)
test_query_2 = "What does the head of Medusa represent in the Sicilian triskeles emblem?"
print("\nTesting Query 2...")
advanced_rag_retrieve(test_query_2, vector_db)

# 測試 3：關於電子質量能量的正規化 (對應 train.csv 的第 3 題)
test_query_3 = "Why is it useful to regularize the mass-energy of an electron with a finite radius?"
print("\nTesting Query 3...")
advanced_rag_retrieve(test_query_3, vector_db)