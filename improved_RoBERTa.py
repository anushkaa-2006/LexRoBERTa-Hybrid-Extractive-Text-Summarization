import nltk
import torch
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from bert_score import score

# Load dataset (CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

# Load RoBERTa tokenizer and fine-tuned model (extractive summarization)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

# Select 100 random documents
sample_documents = random.sample(list(dataset), 100)

# Store reference summaries and generated summaries
references = []
candidates = []

def extractive_summary(text, num_sentences=4):
    """Selects key sentences from text using RoBERTa importance scoring."""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # Return full text if it's already short

    # Encode sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Get model outputs (logits)
    outputs = model(**inputs).logits.squeeze()

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