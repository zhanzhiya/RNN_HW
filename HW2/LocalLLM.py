import torch
import pandas as pd
from sklearn.model_selection import train_test_split # 🌟 補上這個缺失的 import
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import gc

# 釋放記憶體
torch.cuda.empty_cache()
gc.collect()

# ================= CONFIGURATION =================
# 🚨 警告：上傳 GitHub 繳交作業前，請務必把這行清空！
HF_TOKEN = "hf_xxxxxx" 

print("Authenticating with Hugging Face...")
login(token=HF_TOKEN)

GEN_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
DETECTOR_PATH = "./saved_model_bert-base-cased" 
NUM_ATTACKS = 5  # 🌟 作業要求 5-10 篇
# =================================================

# 1. 載入並準備資料
print("Loading data...")
df = pd.read_csv("train_v2_drcat_02.csv") 
df = df[['text', 'label']]
df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)

X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, shuffle=True
)

# 篩選出真實人類寫的文章 (Label 0) 來進行攻擊
human_val_texts = X_val[y_val == 0].tolist()

# 2. 載入生成模型 (Llama-3)
print("\nLoading LLM for generation...")
generator = pipeline(
    "text-generation",
    model=GEN_MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float16}, 
    device_map="auto" 
)

# 3. 載入防禦模型 (BERT)
print(f"Loading BERT Detector from {DETECTOR_PATH}...")
detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to("cuda")
detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)

print("\n" + "="*50)
print(f"🚀 STARTING ADVERSARIAL ATTACK (Testing {NUM_ATTACKS} samples)")
print("="*50)

successful_attacks = 0

# 4. 執行攻擊迴圈 (🌟 這裡幫你把 5 篇測試跑起來)
for i in range(NUM_ATTACKS):
    print(f"\n--- [ Sample {i+1} / {NUM_ATTACKS} ] ---")
    human_text = human_val_texts[i]
    
    # 叫 Llama-3 改寫
    prompt = [
        {"role": "system", "content": "You are a highly advanced text rewriting engine. Your ONLY job is to output the rewritten text. You MUST NOT use any conversational filler, greetings, or prefixes like 'Here is the rewritten essay:'. Output EXACTLY and ONLY the rewritten text."},
        {"role": "user", "content": f"Rewrite the following essay to make it sound like it was written by a high school student.\n\n{human_text}"} ]
     
    output = generator(
        prompt, 
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.7
    )
    
    # 擷取生成的文字
    generated_text = output[0]['generated_text']
    if isinstance(generated_text, list):
        generated_text = generated_text[-1]['content']
        
    print(f"Generated Attack Text (Snippet): {generated_text[:150]}...")
    
    # 用 BERT 檢測
    inputs = detector_tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        logits = detector(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)
        
    ai_prob = probabilities[0][1].item()
    print(f"Detector AI Confidence: {ai_prob:.4f}")
    
    if ai_prob < 0.5:
        print("🎯 SUCCESS! The LLM fooled the detector (Classified as Human).")
        successful_attacks += 1
    else:
        print("🛡️ FAILED. The detector caught the AI (Classified as AI).")

print("\n" + "="*50)
print(f"🏁 Attack Summary: {successful_attacks} out of {NUM_ATTACKS} attacks succeeded!")
print("="*50)