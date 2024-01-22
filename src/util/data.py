import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

sdohs = ['community_present', 'community_absent', 'education', 'economics','environment']
sbdhs = ['alcohol','tobacco','drug']
all_sd = sdohs + sbdhs

class Data:
    def __init__(self, sdoh, path, train_size) -> None:
        if sdoh in sdohs:
            sdoh = f'sdoh_{sdoh}'
        elif sdoh in sbdhs:
            sdoh = f'behavior_{sdoh}'
        else:
            print('Please enter a valid social determinant to extract!')
            exit(1)
            
        dataset = pd.read_csv(path)
        dataset = dataset.dropna()
        self.text_data = dataset["text"].tolist()
        self.sdoh_data = dataset[sdoh].tolist()
        self.max_seq_len = 128 #TODO: Check this with sponsor (calculated as length greater than 90% of entries)
        self.train_size = train_size

    def encode_data(self,model):
        x_train, x_val, y_train, y_val = train_test_split(self.text_data, self.sdoh_data, random_state=0, train_size = self.train_size, stratify=self.sdoh_data)
        train_encodings = model.tokenizer(x_train, truncation=True, padding='max_length', max_length=self.max_seq_len, return_tensors='pt')
        val_encodings = model.tokenizer(x_val, truncation=True, padding='max_length', max_length=self.max_seq_len, return_tensors='pt')
        self.train_dataset = DataLoader(
            train_encodings,
            y_train
        )
        self.val_dataset = DataLoader(
            val_encodings,
            y_val
        )


class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        try:
            # Retrieve tokenized data for the given index
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # Add the label for the given index to the item dictionary
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None

    def __len__(self):
        return len(self.labels)