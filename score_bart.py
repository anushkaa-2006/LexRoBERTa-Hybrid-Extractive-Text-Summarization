import nltk 
import torch
import random
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from bert_score import score

# Load dataset (CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

# Load BART tokenizer and model (pretrained extractive summarization)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Select 100 random documents
random.seed(42)  # For reproducibility
sample_documents = random.sample(list(dataset), 100)

# Store reference summaries and generated summaries
references = []
candidates = []

def extractive_summary(text, min_length=30, max_length=100):
    """Generates a summary using a pre-trained BART model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=min_length, max_length=max_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate summaries and compute scores
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
print("\n\nImproved BERTScore Results:")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1:.4f}")