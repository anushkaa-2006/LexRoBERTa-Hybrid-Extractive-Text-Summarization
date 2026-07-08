import nltk
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import random

# Load dataset
dataset = load_dataset('xsum')

# Get a random instance
random_instance = random.choice(dataset['train'])

# Extract text
text = random_instance['document']
print("\nOriginal Data:\n", text)

# Tokenize text into sentences
sentences = sent_tokenize(text)

# Load RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Encode sentences
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Get model outputs (logits)
with torch.no_grad():
    outputs = model(**inputs).logits

# Convert logits to probabilities
probs = torch.softmax(outputs, dim=1)[:, 1]  # Importance score for each sentence

# Always include the first sentence for context
selected_indices = [0]  # Ensures introduction is included

# Select top-k most important sentences (excluding first since it's already included)
k = min(3, len(sentences) - 1)  # Ensure we don't exceed sentence count
top_indices = torch.argsort(probs[1:], descending=True)[:k] + 1  # Offset since we excluded first

# Combine selected indices and sort by their original order
final_indices = sorted(selected_indices + top_indices.tolist())

# Generate summary
summary_sentences = [sentences[i] for i in final_indices]
summary = " ".join(summary_sentences)

print("\nGenerated Summary:\n", summary)