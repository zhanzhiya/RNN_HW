import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# ==========================================
# 1. 讀取 train.csv 並精煉為「問題 + 正確解答」
# ==========================================
print("Loading train.csv as Clean Knowledge Base...")
train_df = pd.read_csv('train.csv')

raw_documents = []
for index, row in train_df.iterrows():
    # 1. 抓出題目
    question = row['prompt']
    
    # 2. 根據 answer 欄位，直接抓出那一個正確選項的「文字內容」
    correct_option_letter = row['answer']
    correct_option_text = row[correct_option_letter]
    
    # 3. 乾淨俐落的組合：只有問題跟正確答案
    text = f"Question: {question}\nCorrect Answer: {correct_option_text}"
    
    # 建立 LangChain Document
    raw_documents.append(Document(page_content=text, metadata={"id": str(row['id'])}))

print(f"Successfully created {len(raw_documents)} clean documents from train.csv.")

# ==========================================
# 2. 切塊與建立向量資料庫
# ==========================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 確保資料夾存在
os.makedirs("./chroma_db_A", exist_ok=True)
os.makedirs("./chroma_db_B", exist_ok=True)

# Method A: 固定大小切塊 (500 tokens / overlap 50)
print("\nBuilding Strategy A Index...")
splitter_a = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
docs_a = splitter_a.split_documents(raw_documents)
vector_db_a = Chroma.from_documents(
    documents=docs_a, embedding=embeddings, persist_directory="./chroma_db_A", collection_name="strategy_A"
)

# 觀察 Method A 的切塊
print("\n--- Method A Sample Chunks ---")
for i, doc in enumerate(docs_a[:2]): 
    print(f"Chunk {i}: [{doc.page_content}]")

# Method B: 語意切塊 (1000 tokens / overlap 200)
print("\nBuilding Strategy B Index...")
splitter_b = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs_b = splitter_b.split_documents(raw_documents)
vector_db_b = Chroma.from_documents(
    documents=docs_b, embedding=embeddings, persist_directory="./chroma_db_B", collection_name="strategy_B"
)

# 觀察 Method B 的切塊
print("\n--- Method B Sample Chunks ---")
for i, doc in enumerate(docs_b[:2]): 
    print(f"Chunk {i}: [{doc.page_content}]")

print("\n🎉 Part 1 Indexing Complete! 高純度知識庫已成功建立。")