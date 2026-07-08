# 📝 LexRoBERTa: Hybrid Extractive Text Summarization using LexRank and RoBERTa

LexRoBERTa is a hybrid extractive text summarization project that combines the graph-based **LexRank** algorithm with **RoBERTa contextual embeddings** to generate high-quality summaries. The project compares multiple summarization approaches and evaluates their performance using standard NLP evaluation metrics.

---

## 📌 Project Overview

Automatic text summarization has become increasingly important due to the rapid growth of digital information. This project proposes a hybrid approach that combines lexical similarity and contextual semantic understanding to improve extractive summarization quality.

The generated summaries are evaluated and compared with existing methods including TF-IDF, BERT, RoBERTa, ChatGPT, Gemini, and Blackbox AI summaries.

---

## ✨ Features

- Hybrid LexRank + RoBERTa summarization
- Traditional TF-IDF summarization
- BERT-based summarization
- RoBERTa-based summarization
- Comparative evaluation of multiple summarization techniques
- Automatic scoring using standard NLP metrics
- Performance comparison using Excel reports

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning & NLP
- Hugging Face Transformers
- RoBERTa
- BERT
- NLTK
- Scikit-learn
- NumPy
- Pandas

### Evaluation Metrics
- ROUGE
- BLEU
- METEOR
- BERTScore

---

## 📂 Project Structure

```
LexRoBERTa/
│
├── bert.py
├── RoBERTa.py
├── improved_RoBERTa.py
├── hybrid.py
├── project.py
├── sample.py
├── try.py
│
├── private_RoBERTa.py
│
├── score_bert.py
├── score_RoBERTa.py
├── score_blackbox.py
├── score_chatgpt.py
├── score_gemini.py
├── score_bart.py
├── score_TF-IDF.py
│
├── blackbox_summary.csv
├── chatgpt_summaries.csv
├── gemini_summary.csv
├── TF-IDF-summary.csv
├── cnn_dailymail_articles_with_summary.csv
│
├── ETS-score.xlsx
├── CPE_Report_Final.pdf
├── CPE_report.docx
│
└── README.md
```

---

## 📊 Dataset

The project uses the **CNN/DailyMail** dataset containing news articles and their reference summaries for evaluating summarization quality.

---

## 🚀 Methodology

1. Load dataset
2. Preprocess text
3. Generate sentence embeddings using RoBERTa
4. Compute sentence similarity
5. Rank sentences using LexRank
6. Generate extractive summary
7. Compare with other summarization techniques
8. Evaluate using NLP metrics

---

## 📈 Models Compared

- TF-IDF
- BERT
- RoBERTa
- Improved RoBERTa
- Hybrid LexRoBERTa
- ChatGPT
- Gemini
- Blackbox AI

---

## 📊 Evaluation Metrics

The summarization models are evaluated using:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU Score
- METEOR Score
- BERTScore

Evaluation results are available in the generated Excel reports.

---

## ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/anushkaa-2006/LexRoBERTa-Hybrid-Extractive-Text-Summarization.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the hybrid summarization model:

```bash
python hybrid.py
```

Run evaluation scripts:

```bash
python score_RoBERTa.py
python score_bert.py
python score_gemini.py
python score_chatgpt.py
```

---

## 📌 Applications

- News Summarization
- Research Paper Summarization
- Legal Document Summarization
- Educational Content Summarization
- Business Reports
- Article Compression

---

## 🚀 Future Improvements

- Fine-tuned transformer models
- Abstractive summarization support
- PDF document summarization
- Web application interface
- Multi-document summarization
- Multilingual support

---

## 👩‍💻 Developer

**Anushkaa Pawar**

Computer Engineering Student

GitHub: https://github.com/anushkaa-2006

---

## 📄 License

This project is intended for educational and research purposes.
