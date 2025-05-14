from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
df = pd.read_csv("data/logs_indexed.csv")
embeddings = np.load("models/log_embeddings.npy")
model = SentenceTransformer('all-MiniLM-L6-v2')

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/chat")
def chat_with_logs(request: ChatRequest):
    query_emb = model.encode([request.query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = sims.argsort()[-request.top_k:][::-1]
    results = [
        {
            "timestamp": str(df.iloc[i]["timestamp"]),
            "log": str(df.iloc[i]["log"]),
            "score": float(sims[i])  # Convert numpy.float32 to Python float
        }
        for i in top_idx
    ]
    return {"results": results}