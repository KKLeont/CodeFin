import torch
import math
import pandas as pd

def get_position_embedding(value, idx, d):
    embedding = torch.zeros(d)
    embedding[::2] = torch.sin(value / (10000 ** (torch.arange(0, d, 2).float() / d)))
    embedding[1::2] = torch.cos(value / (10000 ** (torch.arange(1, d, 2).float() / d)))
    return embedding

def get_mean_std_emb(ticker):
    if isinstance(ticker, str):
        ticker = [ticker]
    
    df = pd.read_csv('./dataset/mean_std.csv')
    embedding_dim = 32
    all_embeddings = []

    for tick in ticker:
        stock_row = df[df['Ticker'] == tick]

        if stock_row.empty:
            print(f"Ticker {tick} not found in the dataset.")
            continue

        result = stock_row[['Mean_O', 'Std_O', 'Mean_H', 'Std_H', 'Mean_L', 'Std_L', 'Mean_C', 'Std_C']].values.flatten()

        embeddings = []
        for i, value in enumerate(result):
            embeddings.append(get_position_embedding(value, i, embedding_dim))

        embeddings_tensor = torch.stack(embeddings)
        all_embeddings.append(embeddings_tensor)

    if len(all_embeddings) == 0:
        return torch.zeros(0)

    return torch.stack(all_embeddings)
