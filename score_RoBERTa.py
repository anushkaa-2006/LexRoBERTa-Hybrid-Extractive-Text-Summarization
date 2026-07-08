import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import numpy as np
from bert_score import score

# Load tokenizer and model explicitly
model_name = "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extractive_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True) #pt-PyTourch tenser
    summary_ids = model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True) #higher penalty = shorter summaries
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load CNN/Daily Mail dataset
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0", split="test")
sample_documents = cnn_dailymail.select(range(100))  # Select 100 articles

# Lists to store references and candidates
references = []
candidates = []

# Process each document
for doc in sample_documents:
    text = doc["article"]
    reference_summary = doc["highlights"]
    generated_summary = extractive_summary(text)

    references.append(reference_summary)
    candidates.append(generated_summary)

# Compute BERTScore
P, R, F1 = score(candidates, references, model_type="roberta-base", lang="en")

# Compute average scores
avg_precision = np.mean(P.numpy())
avg_recall = np.mean(R.numpy())
avg_f1 = np.mean(F1.numpy())

# Print final results
print("\n\nRoBERTa-Based Extractive Summarization Results:")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1:.4f}")