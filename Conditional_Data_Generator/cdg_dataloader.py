import torch
import random
import pandas as pd
import numpy as np
from scipy.stats import linregress

class MyDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets, codes, timestamps, labels, length):
        self.datasets = datasets
        self.codes = codes
        self.timestamps = timestamps
        self.labels = labels
        self.length = length

    def get_trend_label(self, window_prices, dataset_std):
        if window_prices.shape != (30,4):
            return "Invalid_Window"
        
        close_prices = window_prices[:,3]
        if not isinstance(window_prices, np.ndarray):
            window_prices = np.array(window_prices)
        if window_prices.size == 0:
            return "Invalid_Window"

        try:
            close_prices = window_prices 
            if len(close_prices) < 1:
                return "Empty_Close"
            last_price = close_prices[-1] 
        except IndexError:
            return "Index_Error"
        
        x = np.arange(30)
        try:
            slope, _, _, p_value, _ = linregress(x, close_prices)
        except:
            slope, p_value = 0.0, 1.0
        
        window_mean = np.mean(close_prices)
        window_std = np.std(close_prices)
        last_price = close_prices[-1]
   
        price_changes = np.diff(close_prices)
        max_single_day = np.max(np.abs(price_changes)) if len(price_changes)>0 else 0
        is_extreme = (max_single_day > 3*dataset_std)  
        
        if is_extreme:
            trend = "Extreme"
        elif slope > 0 and p_value < 0.05 and last_price > (window_mean + 0.6*window_std):
            trend = "Uptrend"
        elif slope < 0 and p_value < 0.05 and last_price < (window_mean - 0.6*window_std):
            trend = "Downtrend"
        else:
            trend = "Volatile"
        
        return trend
    
    def choosen_std_get(self, code):
        std_file = pd.read_csv('./dataset/mean_std.csv')
        std = std_file.loc[std_file['Ticker'] == code, 'Std_C'].values
        return std

    def __getitem__(self, idx):
        dataset_idx = random.randint(0, len(self.datasets) - 1)
        dataset_choice = self.datasets[dataset_idx]
        code = self.codes[dataset_idx]
        timestamp_choice = self.timestamps[dataset_idx]
        label_choice = self.labels[dataset_idx]
        dataset_length = len(dataset_choice) - self.length
        random_idx = random.randint(0, dataset_length - 1)
        data = dataset_choice[random_idx: random_idx + self.length]
        timestamp = timestamp_choice[random_idx: random_idx + self.length]
        label = label_choice[random_idx: random_idx + self.length]
        # print('data: ', data.shape)
        choosen_std = self.choosen_std_get(code)
        trend = self.get_trend_label(data,choosen_std)
        return data, code, timestamp, label, trend
    
    def __len__(self):
        total_length = sum(len(dataset) - self.length for dataset in self.datasets)
        return total_length

class Normalizer:
    def __init__(self):
        pass

    def __call__(self, x):
        shape = x.shape
        for i in range(shape[1]):
            mean = x[:,i].mean()
            std = x[:,i].std()
            x[:,i] = (x[:,i] - mean) / std
        return x

def get_position_embedding(value, d):
    value = value.view(-1, 1)  
    embedding = torch.zeros(value.shape[0], d)
    div_term = 10000 ** (torch.arange(0, d, 2).float() / d)

    embedding[:, 0::2] = torch.sin(value / div_term)
    embedding[:, 1::2] = torch.cos(value / div_term)

    return embedding

class MyDataloader:
    def __init__(self, file_paths, batch_size, length, num_workers=4):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.length = length
        self.num_workers = num_workers
        self.datasets, self.codes, self.timestamps, self.labels = self._load_datasets_and_codes(file_paths)

    def _load_datasets_and_codes(self, file_paths):
        datasets = []
        codes = []
        timestamps = []
        labels = []
        embedding_dim = 32
        for file_path in file_paths:
            data = pd.read_csv(file_path)
            timestamp = data.iloc[:,0]
            timestamp = pd.to_datetime(timestamp).astype(int) / 10**9
            timestamp_tensor = torch.tensor(timestamp.values, dtype=torch.float32)
            timestamp_embedding = get_position_embedding(timestamp_tensor, embedding_dim)
            label = data.iloc[:, -1]
            label_tensor = torch.tensor(label.values, dtype=torch.float32)
            code = data.iloc[0, 1] 
            data = data.iloc[:, 2:-1] 
            data_tensor = torch.tensor(data.values, dtype=torch.float32)
            # data_normalized = Normalizer()(data_tensor)
            datasets.append(data_tensor)
            codes.append(code)
            timestamps.append(timestamp_embedding)
            labels.append(label_tensor)
        return datasets, codes, timestamps, labels

    def get_dataloader(self):
        combined_dataset = MyDatasets(self.datasets, self.codes, self.timestamps, self.labels, self.length)
        dataloader = torch.utils.data.DataLoader(dataset=combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_fn)
        return dataloader
    
    def _collate_fn(self, batch):
        data, code, timestamp, label, trend = zip(*batch)
        # print('trend: ',len(trend))
        # print('data: ',len(data))
        return torch.stack(data), code, torch.stack(timestamp), torch.stack(label), trend

    def get_input_size(self):
        input_size = pd.read_csv(self.file_paths[0]).shape[1] - 3  
        return input_size
