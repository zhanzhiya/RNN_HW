import time
import requests
import json
import warnings
import pandas as pd
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")

# ==========================================
# 1. 載入資料庫與模型
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

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading Cross-Encoder: {RERANK_MODEL_NAME} on CUDA...")
reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')

def advanced_rag_retrieve(query, db, top_k_retrieval=20, top_k_rerank=3):
    initial_docs = db.similarity_search(query, k=top_k_retrieval)
    doc_texts = [d.page_content for d in initial_docs]
    pairs = [[query, doc_text] for doc_text in doc_texts]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(initial_docs, scores))
    scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs_sorted[:top_k_rerank]]

# ==========================================
# 2. Ollama 呼叫函數
# ==========================================
def query_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        return response.json()['response']
    except Exception:
        return "Error"

# ==========================================
# 3. 跑 50 題自動化評估 (Evaluation)
# ==========================================
def run_evaluation(csv_path="train.csv", num_questions=50):
    print(f"\n" + "="*60)
    print(f"🚀 開始執行 Part 3: 批量評估與計算準確率 (測試 {num_questions} 題)")
    print("="*60)

    df = pd.read_csv(csv_path).head(num_questions)
    correct_count = 0
    total_time = 0

    for index, row in df.iterrows():
        question = row['prompt']
        options = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\nE) {row['E']}"
        ground_truth = str(row['answer']).strip().upper()

        # 1. 檢索出 Top-3 高純度上下文
        retrieved_docs = advanced_rag_retrieve(question, vector_db)
        context_text = "\n\n".join([d.page_content for d in retrieved_docs])

        # 2. 修改後的 System Prompt：加入「不知道」規則與「單一字母」要求
        prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful science assistant. Answer the multiple-choice question based ONLY on the context provided.

### STRICT RULES:
1. If the answer is not in the context, state "I do not know".
2. If the answer is present, output EXACTLY ONE LETTER (A, B, C, D, or E) corresponding to the correct option. 
3. Do NOT provide any explanation, sentences, or punctuation.

Context:
{context_text}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
Options:
{options}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

        # 3. 呼叫 LLM
        t_start = time.time()
        raw_ans = query_ollama(prompt).strip()
        llm_time = time.time() - t_start
        total_time += llm_time

        # 4. 判讀 LLM 回答
        # 邏輯：如果回答包含 "I do not know"，預測設為 "UNKNOWN"
        # 否則，抓取第一個英文字母
        if "know" in raw_ans.lower():
            prediction = "UNKNOWN"
        else:
            prediction = raw_ans[0].upper() if len(raw_ans) > 0 else "X"
        
        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_count += 1
            icon = "✅"
        else:
            icon = "❌"

        print(f"[{index+1:02d}/{num_questions}] Time: {llm_time:.2f}s | Truth: {ground_truth} | Pred: {prediction} | {icon}")

    # ==========================================
    # 4. 結算成績單
    # ==========================================
    accuracy = (correct_count / num_questions) * 100
    avg_time = total_time / num_questions
    print("\n" + "="*60)
    print("🏆 Part 3 Evaluation Complete!")
    print(f"Total Questions Evaluated : {num_questions}")
    print(f"Total Correct Answers     : {correct_count}")
    print(f"Average Generation Time   : {avg_time:.2f} seconds/question")
    print(f"🔥 Final Accuracy         : {accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_evaluation(csv_path="train.csv", num_questions=50)