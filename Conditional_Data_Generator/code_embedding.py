import numpy as np
import torch
import torch.nn as nn

ticker_list=[
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "PEP", "COST",
            "AVGO", "CSCO", "ADBE", "NFLX", "PYPL", "INTC", "CMCSA", "TXN", "QCOM", "AMGN",
            "HON", "BKNG", "INTU", "AMD", "ISRG", "ADI", "MU", "MDLZ", "VRTX", "REGN",
            "KLAC", "LRCX", "PANW", "ADP", "ABNB", "CRWD", "SNPS", "SBUX", "MELI", "CTAS",
            "MAR", "NXPI", "CSX", "CEG", "ORLY", "MRVL", "ROP", "PCAR", "ADSK", "CPRT",
            "MNST", "MCHP", "ROST", "TTD", "AEP", "FTNT", "TEAM", "MRNA", "KDP", "DXCM",
            "ILMN", "DDOG", "ZS", "DLTR", "VRSK", "AMAT", "BKR", "CSGP", "SIRI", "CDW",
            "AZN", "WDAY", "IDXX", "ALGN", "CDNS", "ODFL", "FAST", "EXC", "FISV", "ATVI",
            "PDD", "LCID", "BIDU", "JD", "NTES", "EBAY", "CTSH", "SGEN", "BMRN", "MTCH",
            "CHTR", "DOCU", "ASML", "BIIB", "WDAY", "TSCO", "ZM", "VRSN", "SPLK", "WBA", "avg"
        ]

class code_emb:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.stock_codes = ticker_list

        self.embedding_dim = 32
        self.stock_code_to_index = {code: idx for idx, code in enumerate(self.stock_codes)}
        self.embedding_matrix = nn.Embedding(len(self.stock_codes), self.embedding_dim)

    def get_stock_embedding(self):
        if isinstance(self.stock_code, str): 
            self.stock_code = [self.stock_code]
        embeddings = []
        for code in self.stock_code:
            idx = self.stock_code_to_index[code]
            embedding = self.embedding_matrix(torch.tensor([idx], dtype=torch.long))
            embeddings.append(embedding)
        return torch.stack(embeddings)


