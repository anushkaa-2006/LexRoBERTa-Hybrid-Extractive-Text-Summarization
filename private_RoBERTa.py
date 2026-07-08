import nltk
import torch
import random
import numpy as np
from transformers import RobertaTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from bert_score import score
from huggingface_hub import login

# Login to Hugging Face
login(token="--------") #enter your hugging face token

# Load dataset (CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

# Load RoBERTa tokenizer and model for summarization
model_name = "Salesforce/mixtral-8x7b-sft-roberta-large"  # Correct model
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Select 100 random documents
random.seed(42)
sample_documents = random.sample(list(dataset), 100)

# Store reference summaries and generated summaries
references = []
candidates = []

def extractive_summary(text, min_length=30, max_length=100):
    """Generates a summary using a RoBERTa-based model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
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
print("\n\nRoBERTa-Based Extractive Summarization Results:")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1:.4f}")


