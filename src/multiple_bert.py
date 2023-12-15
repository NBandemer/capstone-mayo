from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

import sys
import datetime
import os
import warnings

from helper import *

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

dataset = pd.read_csv("data\clean\PREPROCESSED-NOTES.csv")

text_data = dataset["text"].to_list()
sdoh_data = dataset["behavior_alcohol"].to_list()

X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size = .8, stratify=sdoh_data)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0, test_size = .01)

max_seq_length = 100 #512

# Truncate and tokenize your input data
train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

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

train_dataset = DataLoader(
    train_encodings,
    y_train
)

val_dataset = DataLoader(
    val_encodings,
    y_val
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

tensor_logs = f'./logs/tensor_logs/logs_{timestamp}'
os.makedirs(tensor_logs, exist_ok=True)

epoch_logs = f'./logs/epoch_logs/logs_{timestamp}'
os.makedirs(epoch_logs, exist_ok=True)

training_args = TrainingArguments(
    output_dir=epoch_logs,
    logging_dir=tensor_logs,
    save_strategy='epoch',
    num_train_epochs=5,
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,
    weight_decay=1e-5,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                 
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,    
    compute_metrics=compute_metrics        
)

trainer.train()
trainer.evaluate()

# convert to eval results to csv
latest_checkpoint = get_latest_checkpoint(epoch_logs)
json_path = os.path.join(latest_checkpoint, 'trainer_state.json')
save_metrics_to_csv(json_path, 'eval_metric.csv')

plot_metrics_from_csv('eval_metric.csv', 'graphs')

# Saving & Loading the model<br>
save_directory = "saved_models/bert" 
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)