# 🛡️ Social Media Hate Speech Detection

A deep learning-based multi-class hate speech detection system using **Bidirectional LSTM with Attention Mechanism**, achieving **87% validation accuracy** on Twitter social media data.

> 📄 This project is based on our published research paper:  
> **"A Novel Deep Learning Approach for Hate Speech Detection on Social Platforms"**  
> *Hemalatha A N, Impana K P, Vikhyath K B, Pavana M, Nagadeepa S Shetti*

---

## 📌 Project Overview

Social media platforms face growing challenges with hate speech and offensive content. This project presents a robust NLP pipeline to automatically classify social media posts into:

| Class | Description |
|-------|-------------|
| 🟢 Neutral | Clean, non-offensive content |
| 🟡 Offensive | Offensive but not targeted hate |
| 🔴 Hate Speech | Explicit hate targeting groups |

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | **87%** |
| Training Accuracy | 97% |
| Offensive F1-Score | 0.93 |
| Neutral F1-Score | 0.86 |
| Hate Speech F1-Score | 0.71 |
| AUC (Neutral) | 0.94 |
| AUC (Offensive) | 0.91 |
| AUC (Hate Speech) | 0.89 |

---

## 🧠 Model Architecture

```
Input Layer (Max Length: 50 tokens)
        ↓
Embedding Layer (FastText, 300D, 25K vocab)
        ↓
Bidirectional LSTM 1 (128 units, tanh)
        ↓
Dropout Layer 1 (rate=0.5)
        ↓
Bidirectional LSTM 2 (64 units, tanh)
        ↓
Dropout Layer 2 (rate=0.5)
        ↓
Self-Attention Layer (8 heads)
        ↓
Dense Layer (64 units, ReLU)
        ↓
Output Layer (3 units, Softmax)
```

---

## 🔧 Tech Stack

- **Language:** Python 3.7+
- **Deep Learning:** TensorFlow 2.x / Keras
- **ML Libraries:** Scikit-learn, NLTK
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Embeddings:** FastText (300D)
- **IDE:** Jupyter Notebook / Google Colab

---

## 📂 Project Structure

```
hate-speech-detection/
│
├── hate_speech_detection.ipynb   # Main notebook (EDA + Model + Evaluation)
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/HemalathaAN/hate-speech-detection.git
cd hate-speech-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the Dataset
Download the hate speech dataset from Kaggle:
- [Twitter Hate Speech Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
- Place the CSV file in the project root directory

### 4. Run the notebook
```bash
jupyter notebook hate_speech_detection.ipynb
```
Or open directly in **Google Colab** for GPU support.

---

## 📊 Key Features

- ✅ Multi-stage NLP preprocessing pipeline
- ✅ Social media specific text cleaning (URLs, mentions, hashtags)
- ✅ Class imbalance handling with class weights
- ✅ Bidirectional LSTM with self-attention
- ✅ Comprehensive evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
- ✅ Ablation study comparing multiple architectures

---

## 📈 Preprocessing Pipeline

1. **Text Normalization** — Lowercase, Unicode (NFKC), contraction expansion
2. **Social Media Cleaning** — Remove URLs, replace @mentions, segment hashtags
3. **Tokenization** — NLTK Tweet Tokenizer
4. **Stop Word Filtering** — Custom list preserving negations
5. **Sequence Padding** — Fixed length of 50 tokens

---

## 👩‍💻 Author

**Hemalatha A N**  
Data Analyst | Python · SQL · Power BI · Tableau | ML & NLP  
🔗 [LinkedIn](https://www.linkedin.com/in/hemalatha-an)  
🐙 [GitHub](https://github.com/HemalathaAN)

---

## 📜 Research Paper

This project is based on the published paper:

> *Hemalatha A N et al., "A Novel Deep Learning Approach for Hate Speech Detection on Social Platforms", JSS Academy of Technical Education, Bengaluru, 2025.*

---

## 📄 License

This project is for academic and research purposes.

