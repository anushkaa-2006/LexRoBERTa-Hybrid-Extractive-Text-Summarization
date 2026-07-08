from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity

dataset = load_dataset('cnn_dailymail', '3.0.0')  # Changed dataset to CNN/DailyMail

random_instance = random.choice(dataset['train'])  # Random instance
# random_instance is now a dictionary that contains information about the article

random_text = random_instance['article']  # Extract article text
print("\nOriginal Data:\n", random_text)

# Loads pre-trained BERT tokenizer and BERT model.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize the whole document
inputs = tokenizer(random_text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)  # Pass the tokenized input into the BERT model
doc_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings (document embedding) [SEP]

# Document splitting based on '.' (full stop)
sentences = random_text.split('.')

# Get embeddings for each sentence
sentence_embeddings = []
for sentence in sentences:
    sentence = sentence.strip()  # Removes extra spaces
    if len(sentence) > 0:  # Skip empty sentences
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings [SEP]
        sentence_embeddings.append(sentence_embedding)

# Convert the list of sentence embeddings to a tensor (multi-dimensional array representing sentences in numerical form)
sentence_embeddings = torch.cat(sentence_embeddings, dim=0)  # cat() joins multiple tensors

# Calculate cosine similarity between document embedding and sentence embeddings
cosine_similarities = cosine_similarity(doc_embedding.detach().numpy(), sentence_embeddings.detach().numpy())  # numpy() converts tensor into NumPy array

# Rank sentences based on similarity to the document
# zip function combines the similarity scores with their corresponding sentences into pairs (tuples)
ranked_sentences = sorted(zip(cosine_similarities[0], sentences), reverse=True, key=lambda x: x[0])

# Select the top 5 ranked sentences
top_5_ranked_sentences = [sentence for _, sentence in ranked_sentences[:5]]

# Ordered sentences
top_5_ordered_sentences = [sentence for sentence in sentences if sentence in top_5_ranked_sentences]

# Create a summary by appending the selected sentences in sequence
summary = " ".join([sentence.strip() for sentence in top_5_ordered_sentences])

print("\n \nSummary : \n")
print(summary, "\n")  # Display summary
