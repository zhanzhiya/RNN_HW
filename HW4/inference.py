import os
# 把模型下載路徑改到D槽專案資料夾下
os.environ["HF_HOME"] = "D:/zhiya/hf_cache"
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset

# ==========================================
# 1. 模型與量化設定 (符合 VRAM 限制)
# ==========================================
model_id = "llava-hf/llava-1.5-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading LLaVA in 4-bit...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
from transformers import LlavaProcessor, CLIPImageProcessor, LlamaTokenizer

print("正在手動拼裝 Processor...")
tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(model_id)
processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
print("Model loaded successfully!")

# ==========================================
# 2. 準備 ChartQA 測試資料 (Zero-Shot Testing)
# ==========================================
print("Loading ChartQA dataset samples...")
# 1. 先載入完整的驗證集
full_dataset = load_dataset("HuggingFaceM4/ChartQA", split="val")

# 2. 利用 select 功能，刻意跳著挑選索引值為 x 的資料
dataset = full_dataset.select([5, 10, 2])

# ==========================================
# 3. 執行推論與觀察 (Inference)
# ==========================================
print("\n" + "="*50)
print("Starting Zero-Shot Inference Baseline")
print("="*50)

for i, data in enumerate(dataset):
    image = data['image']
    # ChartQA 資料集已經準備好問題和標準答案了
    question = data['query']
    ground_truth = data['label'] 
    
    # 組裝 LLaVA 專用的 Prompt 格式 (必須包含 <image> 標籤)
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    
    # 處理輸入並送入 GPU
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # 生成回答
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # 擷取 ASSISTANT 後面的純回答部分
    answer = output.split("ASSISTANT:")[-1].strip()
    
    # 印出結果報告紀錄
    print(f"\n[Test Image {i+1}]")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Model's 'Before' Answer: {answer}")
    print("-" * 50)
    
    # 將圖片存下來，報告時做前後對比截圖
    image.save(f"hw4_baseline_image_{i+1}.png")
