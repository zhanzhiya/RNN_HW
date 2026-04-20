import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
MODEL_NAME = "bert-base-cased" 
#MODEL_NAME = "bert-large-cased" 

MAX_LEN = 512
BATCH_SIZE = 64  # RTX 4090 can handle 16-32 for Large, 32-64 for Base
EPOCHS = 5
# =================================================

# 1. 載入資料與切分 (讓 BERT.py 可以獨立運作)
print("Loading data...")
df = pd.read_csv("train_v2_drcat_02.csv") # 確認檔名是否正確
df = df[['text', 'label']]
df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)


# 確保切分方式與 Baseline 完全一致
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, shuffle=True
)

# 2. 準備 Hugging Face Dataset
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})

hf_train = Dataset.from_pandas(train_df)
hf_val = Dataset.from_pandas(val_df)

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

# 新增：Tokenize 後移除不需要的原始文字欄位 (如果你的 DataFrame index 也被帶入，建議一併移除)
tokenized_train = hf_train.map(tokenize_function, batched=True).remove_columns(["text"])
tokenized_val = hf_val.map(tokenize_function, batched=True).remove_columns(["text"])
# 如果有 "__index_level_0__" 這種 pandas 自動產生的欄位，也可以在上述 remove_columns 裡一併寫入。

# 3. Model Setup
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# =================================================
# 新增：自定義計算 ROC-AUC 的 Metric 函式
# =================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 將模型輸出的 logits 轉換為機率值
    probs = softmax(predictions, axis=1)
    # 取出預測為類別 1 (AI 生成) 的機率
    preds_prob_1 = probs[:, 1]
    
    # 計算 ROC-AUC
    auc = roc_auc_score(labels, preds_prob_1)
    return {"roc_auc": auc}
# =================================================

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir=f"./results_{MODEL_NAME}",
    eval_strategy="epoch",  # 在每個 epoch 結束時進行評估
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,  
    logging_steps=50,
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics, # <--- 新增：告訴 Trainer 評估時要用這個函式
)

print(f"Starting training for {MODEL_NAME}...")
trainer.train()

# 6. Evaluation
results = trainer.evaluate()
# ==========================================
# 新增：自動繪製並儲存 Loss 曲線圖
# ==========================================
print("Generating Loss Curve...")

# 從 trainer 的紀錄中抓取訓練與驗證的 loss
log_history = trainer.state.log_history

train_epochs = []
train_loss = []
eval_epochs = []
eval_loss = []

for log in log_history:
    if 'loss' in log and 'epoch' in log:
        train_epochs.append(log['epoch'])
        train_loss.append(log['loss'])
    elif 'eval_loss' in log and 'epoch' in log:
        eval_epochs.append(log['epoch'])
        eval_loss.append(log['eval_loss'])

# 開始畫圖
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_loss, label='Training Loss', marker='o', linestyle='-', color='blue')
plt.plot(eval_epochs, eval_loss, label='Validation Loss', marker='s', linestyle='--', color='red')

plt.title(f'{MODEL_NAME} Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 儲存圖片
plot_filename = f'loss_curve_{MODEL_NAME}.png'
plt.savefig(plot_filename)
print(f"Loss curve saved to {plot_filename}")
# ==========================================
# 這裡印出來的結果現在就會包含 'eval_roc_auc' 了
print(f"Evaluation Results for {MODEL_NAME}: {results}") 

# Save the model for Part 3
model.save_pretrained(f"./saved_model_{MODEL_NAME}")
tokenizer.save_pretrained(f"./saved_model_{MODEL_NAME}")

