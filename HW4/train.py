import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib.pyplot as plt 
print("===== 1. 程式開始執行 =====")

from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
    LlamaTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
print("===== 2. Transformers 載入成功 =====")

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
print("===== 3. PEFT 載入成功 =====")

from datasets import load_dataset
print("===== 4. Datasets 載入成功 =====")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# 1. 載入模型
# =========================================================
model_id = "llava-hf/llava-1.5-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
print("Model loaded successfully!")

tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
image_processor = CLIPImageProcessor.from_pretrained(model_id)

processor = LlavaProcessor(
    tokenizer=tokenizer,
    image_processor=image_processor
)
processor.tokenizer.padding_side = "right"

# =========================================================
# 2. 準備 QLoRA
# =========================================================
print("Preparing model for QLoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# =========================================================
# 3. 載入資料集 (抓前 600 筆)
# =========================================================
print("Loading dataset...")
dataset = load_dataset("HuggingFaceM4/ChartQA", split="train[:600]")

# =========================================================
# 4. 動態資料處理 (On-the-fly Data Collator)
# =========================================================
print("Setting up dynamic collator...")
class MultimodalCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for ex in examples:
            ans = ex["label"][0] if isinstance(ex["label"], list) else ex["label"]
            prompt = f"USER: <image>\n{ex['query']}\nASSISTANT: {ans}"
            texts.append(prompt)
            images.append(ex["image"])

        batch = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

my_collator = MultimodalCollator(processor)

# =========================================================
# 5. Training Arguments
# =========================================================
print("Setting up training...")
training_args = TrainingArguments(
    output_dir="./llava-chartqa-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    optim="adamw_torch",           
    remove_unused_columns=False,   
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    report_to="none",
    gradient_checkpointing=True,
)

# =========================================================
# 6. Trainer
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,      
    data_collator=my_collator,  
)

trainer.floating_point_ops = lambda *args, **kwargs: 0

# =========================================================
# 7. 開始訓練與儲存
# =========================================================
print("\n🚀 開始訓練 QLoRA...\n")
trainer.train()

trainer.save_model("./llava-chartqa-lora-final")
print("\n✅ 訓練完成！模型已儲存！")

# =========================================================
# 8. 自動繪製並儲存 Loss 曲線圖
# =========================================================
print("\n📊 正在繪製 Loss 曲線圖...")
history = trainer.state.log_history

# 直接從訓練紀錄中萃取 loss 和 epoch
epochs = [log["epoch"] for log in history if "loss" in log]
losses = [log["loss"] for log in history if "loss" in log]

if len(losses) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='#FF5733', linewidth=2.5, markersize=8, label="Training Loss")
    
    plt.title("QLoRA Training Loss Curve (ChartQA)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 將圖片存檔到HW4資料夾底下
    save_path = "loss_curve_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 畫圖成功！圖片已自動儲存為：{save_path}")
    
    # 在螢幕上顯示圖片
    plt.show()
else:
    print("⚠️ 找不到 Loss 紀錄，無法畫圖。")

print("🎉 所有任務圓滿結束！")