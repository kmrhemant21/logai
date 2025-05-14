# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os

# os.makedirs("models", exist_ok=True)

# df = pd.read_csv("data/logs.csv")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(df['log'].tolist(), show_progress_bar=True)
# np.save("models/log_embeddings.npy", embeddings)
# df.to_csv("data/logs_indexed.csv", index=False)
# print("Embeddings generated and saved.")
import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel

# Create output directory
os.makedirs("models", exist_ok=True)

# Load logs
df = pd.read_csv("data/logs.csv")

# Load more powerful model and tokenizer
model_name = "thenlper/gte-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Function to compute mean pooled sentence embeddings
def compute_embeddings(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            output = model(**encoded_input)
            # Mean pooling
            attention_mask = encoded_input['attention_mask']
            last_hidden_state = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            mean_pooled = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
            embeddings.append(mean_pooled.cpu().numpy())
    return np.vstack(embeddings)

# Generate embeddings
print("Generating high-quality embeddings...")
embeddings = compute_embeddings(df['log'].tolist())
np.save("models/log_embeddings.npy", embeddings)

# Save updated DataFrame
df.to_csv("data/logs_indexed.csv", index=False)
print("Embeddings generated and saved.")