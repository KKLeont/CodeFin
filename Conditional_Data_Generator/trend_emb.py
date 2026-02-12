import numpy as np
import torch
import torch.nn as nn

class TrendEmb:
    def __init__(self, trend):
        self.trend = trend
        self.trends = ["Uptrend", "Downtrend", "Volatile", "Extreme"]
        self.embedding_dim = 4
        self.trend_to_index = {code: idx for idx, code in enumerate(self.trends)}
        self.embedding_matrix = nn.Embedding(len(self.trends), self.embedding_dim)

    def get_trend_embedding(self):
        if isinstance(self.trend, str): 
            self.trend = [self.trend]

        embeddings = []
        for trend in self.trend:
            idx = self.trend_to_index[trend]
            embedding = self.embedding_matrix(torch.tensor([idx], dtype=torch.long))
            embeddings.append(embedding)
        
        return torch.stack(embeddings)