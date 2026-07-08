from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np

# Load dataset and model
dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')  # Use 'test' split
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to calculate BERTScore for a batch of documents
def calculate_bertscore_batch(documents):
    references = [doc['highlights'] for doc in documents]
    candidates = []

    for random_instance in documents:
        random_text = random_instance['article']  # Extract article text
        sentences = random_text.split('.')
        
        # Get embeddings for each sentence
        sentence_embeddings = []
        inputs = tokenizer(random_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        doc_embedding = outputs.last_hidden_state[:, 0, :]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0:
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                sentence_embedding = outputs.last_hidden_state[:, 0, :]
                sentence_embeddings.append(sentence_embedding)

        # Convert the list of sentence embeddings to a tensor
        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        
        # Calculate cosine similarity between document embedding and sentence embeddings
        cosine_similarities = cosine_similarity(doc_embedding.detach().numpy(), sentence_embeddings.detach().numpy())
        
        # Rank sentences based on similarity to the document
        ranked_sentences = sorted(zip(cosine_similarities[0], sentences), reverse=True, key=lambda x: x[0])
        
        # Select the top 5 ranked sentences
        top_5_ranked_sentences = [sentence for _, sentence in ranked_sentences[:5]]
        
        # Ordered sentences
        top_5_ordered_sentences = [sentence for sentence in sentences if sentence in top_5_ranked_sentences]
        
        # Create a summary
        summary = " ".join([sentence.strip() for sentence in top_5_ordered_sentences])
        candidates.append(summary)

    # Compute BERTScore for the batch
    P, R, F1 = score(candidates, references, model_type="bert-base-uncased", lang="en")
    
    # Average the BERTScore for the batch
    avg_precision = np.mean(P.numpy())
    avg_recall = np.mean(R.numpy())
    avg_f1 = np.mean(F1.numpy())

    return avg_precision, avg_recall, avg_f1

# Convert dataset to a list and select **100** documents
sample_documents = random.sample(list(dataset), 100)

# Calculate BERTScore for the sample batch
avg_precision, avg_recall, avg_f1 = calculate_bertscore_batch(sample_documents)

# Print results
print(f"\n\nBERTScore Results:\nPrecision: {avg_precision:.4f}\nRecall: {avg_recall:.4f}\nF1 Score: {avg_f1:.4f}")
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and model
dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')  # Use 'test' split
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)  # Move model to GPU

# Function to calculate BERTScore for a batch of documents
def calculate_bertscore_batch(documents):
    references = [doc['highlights'] for doc in documents]
    candidates = []

    for random_instance in documents:
        random_text = random_instance['article']  # Extract article text
        sentences = random_text.split('.')

        # Get embeddings for the entire document
        inputs = tokenizer(random_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():  # Disable gradient calculation to speed up inference
            outputs = model(**inputs)
        doc_embedding = outputs.last_hidden_state[:, 0, :]

        # Get embeddings for each sentence (batch processing)
        sentence_embeddings = []
        batch_size = 8  # Process in batches for efficiency
        for i in range(0, len(sentences), batch_size):
            batch_sentences = [s.strip() for s in sentences[i:i + batch_size] if s.strip()]
            if not batch_sentences:
                continue

            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            sentence_embeddings.append(outputs.last_hidden_state[:, 0, :])

        if not sentence_embeddings:
            continue  # Skip if no sentences are extracted

        # Convert to tensor
        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(doc_embedding.cpu().numpy(), sentence_embeddings.cpu().numpy())

        # Rank sentences and select top 3
        ranked_sentences = sorted(zip(cosine_similarities[0], sentences), reverse=True, key=lambda x: x[0])
        top_3_ranked_sentences = [sentence for _, sentence in ranked_sentences[:3]]

        # Ordered sentences
        top_3_ordered_sentences = [sentence for sentence in sentences if sentence in top_3_ranked_sentences]

        # Create summary
        summary = " ".join([sentence.strip() for sentence in top_3_ordered_sentences])
        candidates.append(summary)

    # Compute BERTScore for the batch
    P, R, F1 = score(candidates, references, model_type="bert-base-uncased", lang="en")

    # Average the BERTScore for the batch
    avg_precision = np.mean(P.cpu().numpy())
    avg_recall = np.mean(R.cpu().numpy())
    avg_f1 = np.mean(F1.cpu().numpy())

    return avg_precision, avg_recall, avg_f1

# Convert dataset to a list and select 100 documents
sample_documents = random.sample(list(dataset), 100)

# Calculate BERTScore for the sample batch
avg_precision, avg_recall, avg_f1 = calculate_bertscore_batch(sample_documents)

# Print results
print(f"\n\nBERTScore Results:\nPrecision: {avg_precision:.4f}\nRecall: {avg_recall:.4f}\nF1 Score: {avg_f1:.4f}")
