import nltk
import torch
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from bert_score import score

# Load dataset (CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

# Load RoBERTa tokenizer and model for extractive summarization
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
model_roberta = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

# Load BART tokenizer and model for abstractive summarization
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Select 100 random documents
sample_documents = random.sample(list(dataset), 100)

# Store reference summaries and generated summaries
references = []
candidates = []

def extractive_summary(text):
    """Selects key sentences from text using RoBERTa importance scoring, ensuring the summary is 1/2 the length of the original text."""
    sentences = sent_tokenize(text)
    num_sentences = max(1, len(sentences) // 2)  # Ensure at least one sentence is included
    
    if len(sentences) <= num_sentences:
        return text  # Return full text if it's already short

    # Encode sentences
    inputs = tokenizer_roberta(sentences, padding=True, truncation=True, return_tensors="pt")

    # Get model outputs (logits)
    outputs = model_roberta(**inputs).logits.squeeze()

    # Select sentences dynamically (ensuring balance)
    importance_scores = outputs.tolist()
    ranked_sentences = sorted(zip(importance_scores, sentences), reverse=True)

    # Always include the first sentence for context
    selected_sentences = [sentences[0]]

    # Select top-ranked sentences (excluding the first)
    top_sentences = [s[1] for s in ranked_sentences[1:num_sentences]]
    
    # Maintain original order
    final_summary = " ".join(sorted(selected_sentences + top_sentences, key=lambda s: sentences.index(s)))
    
    return final_summary

def abstractive_summary(text):
    """Generates an abstractive summary using BART model."""
    inputs = tokenizer_bart(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_bart.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)

# Generate summaries and compute scores
for doc in sample_documents:
    text = doc["article"]
    reference_summary = doc["highlights"]
    
    extractive = extractive_summary(text)  # Generate 1/2 summary using RoBERTa
    generated_summary = abstractive_summary(extractive)  # Generate final summary using BART

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
