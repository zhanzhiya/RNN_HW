import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from PIL import Image

print("===== 1. 開始載入模型 =====")

# 1. 基礎模型路徑
base_model_id = "llava-hf/llava-1.5-7b-hf"
# 2. 剛剛訓練好的 LoRA 權重資料夾路徑
adapter_path = "./llava-chartqa-lora-final" 

# 載入 Processor (負責處理圖片和文字)
from transformers import LlavaProcessor, CLIPImageProcessor, LlamaTokenizer

print("正在載入 Processor...")
tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(base_model_id)
processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

# 載入基礎模型 (為了避免 OOM，用 4-bit 載入，並強迫用 auto 分配)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("正在載入 Base Model (這可能需要一分鐘)...")
base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# 把基礎模型跟你的 LoRA 權重合體！
print(f"正在掛載 LoRA Adapter: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("✅ 模型合體完成！準備開始測試！")

# =========================================================
# 準備要測試的圖片和問題 (換成 Part 1 用過的圖片路徑)
# =========================================================
# 這裡是一個 List，裡面放著 Dictionary。可以一直往下加。
test_cases = [
    {
        "image_path": "D:\zhiya\RNN_HW\HW4\hw4_baseline_image_1.png",  # 第一張圖片檔名
        "query": "Which line has the lowest value of 71%?" # 第一張圖片的問題
    },
    {
        "image_path": "D:\zhiya\RNN_HW\HW4\hw4_baseline_image_2.png",  
        "query": "Which indicator remains all time lowest from Dec. 2008 to Sep. 2011?"
    },
    {
        "image_path": "D:\zhiya\RNN_HW\HW4\hw4_baseline_image_3.png",  
        "query":  "What does the blue line represent?"
    },
]

print("\n" + "="*50)
print("🚀 開始進行推論 (Inference) 測試")
print("="*50 + "\n")

for i, test_case in enumerate(test_cases):
    img_path = test_case["image_path"]
    prompt_text = test_case["query"]
    
    print(f"--- 測試案例 {i+1} ---")
    print(f"圖片路徑: {img_path}")
    print(f"使用者問題: {prompt_text}")
    
    try:
        # 開啟圖片
        raw_image = Image.open(img_path).convert("RGB")
        
        # 組合 Prompt (要符合 LLaVA 的固定格式)
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        
        # 預處理
        inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda", torch.float16)
        
        # 讓模型生答案
        output = model.generate(
            **inputs, 
            max_new_tokens=100,  # 最多生成 100 個字
            do_sample=False      # 關閉隨機抽樣，確保每次答案一致
        )
        
        # 解碼答案 (只抓 ASSISTANT: 後面的字)
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        answer = generated_text.split("ASSISTANT:")[-1].strip()
        
        print(f"\n🤖 訓練後的模型回答: \n{answer}\n")
        print("-" * 30 + "\n")
        
    except FileNotFoundError:
        print(f"❌ 找不到圖片檔案: {img_path}，請檢查路徑是否正確！\n")