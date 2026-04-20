import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# 1. 載入與準備資料
# ==========================================
print("Loading data...")
df = pd.read_csv("train_v2_drcat_02.csv")  
df = df[['text', 'label']] 

# 🌟 使用隨機抽樣 5%，設定 random_state 確保每次抽樣結果相同
df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)

# ==========================================
# 2. 進行 EDA (探索性資料分析 - 作業 Part 1)
# ==========================================
print("進行 EDA 分析...")

# 計算字數 (Word Count)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# 計算詞彙豐富度 (相異詞彙數 / 總字數)
def calculate_richness(text):
    words = str(text).split()
    if not words: return 0
    return len(set(words)) / len(words)

df['vocab_richness'] = df['text'].apply(calculate_richness)

# 繪製並儲存圖表
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=df, x='word_count', hue='label', bins=50, ax=axes[0], kde=True)
axes[0].set_title('Word Count Distribution (0: Human, 1: AI)')

sns.histplot(data=df, x='vocab_richness', hue='label', bins=50, ax=axes[1], kde=True)
axes[1].set_title('Vocabulary Richness Distribution')
plt.tight_layout()
plt.savefig('eda_plots.png')
print("EDA 圖表已儲存為 eda_plots.png")

# ==========================================
# 3. 切分資料集
# ==========================================
# 🌟 切分資料時開啟 shuffle，並設定相同的 random_state
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, shuffle=True
)

# ==========================================
# 4. TF-IDF + Logistic Regression Baseline
# ==========================================
print("Training TF-IDF Baseline...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train_tfidf, y_train)

# Evaluation
preds = clf.predict_proba(X_val_tfidf)[:, 1]
auc = roc_auc_score(y_val, preds)
print(f"Baseline TF-IDF ROC-AUC: {auc:.4f}")