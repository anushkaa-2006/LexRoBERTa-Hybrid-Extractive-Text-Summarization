import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from bert_score import score
import numpy as np

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)

# Load the CSV files with new names
cnn_summaries_path = "cnn_dailymail_articles_with_summary_org.csv"  
chatgpt_summaries_path = "TF-IDF_summary.csv"

# Read files
df_cnn = pd.read_csv(cnn_summaries_path)
df_chatgpt = pd.read_csv(chatgpt_summaries_path)

# Ensure the CSVs have expected columns
expected_summary_column = 'Summary'

if expected_summary_column not in df_cnn.columns:
    raise ValueError(f"CNN file must contain column: {expected_summary_column}")
if expected_summary_column not in df_chatgpt.columns:
    raise ValueError(f"ChatGPT file must contain column: {expected_summary_column}")

# Extract first 100 summaries
cnn_summaries = df_cnn[expected_summary_column].tolist()[:100]
chatgpt_summaries = df_chatgpt[expected_summary_column].tolist()[:100]

# Ensure both lists have 100 entries
if len(cnn_summaries) < 100 or len(chatgpt_summaries) < 100:
    raise ValueError("Both summary files must contain at least 100 summaries.")

# Compute BERTScore
P, R, F1 = score(cnn_summaries, chatgpt_summaries, model_type="bert-base-uncased", lang="en")

# Calculate statistics
avg_precision = np.mean(P.cpu().numpy())
min_precision = np.min(P.cpu().numpy())
max_precision = np.max(P.cpu().numpy())

avg_recall = np.mean(R.cpu().numpy())
min_recall = np.min(R.cpu().numpy())
max_recall = np.max(R.cpu().numpy())

avg_f1 = np.mean(F1.cpu().numpy())
min_f1 = np.min(F1.cpu().numpy())
max_f1 = np.max(F1.cpu().numpy())

# Print results
print("\n\nBERTScore Results (CNN DailyMail Articles vs. Gemini Summaries):")
print(f"Precision: Avg={avg_precision:.4f}, Min={min_precision:.4f}, Max={max_precision:.4f}")
print(f"Recall: Avg={avg_recall:.4f}, Min={min_recall:.4f}, Max={max_recall:.4f}")
print(f"F1 Score: Avg={avg_f1:.4f}, Min={min_f1:.4f}, Max={max_f1:.4f}")