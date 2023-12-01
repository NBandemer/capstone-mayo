from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer

import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

import sys
import datetime
import os

num_labels_per_sdoh = [2, 2, 3, 3, 5, 5, 5]

tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
models = [BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_labels) for num_labels in num_labels_per_sdoh]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model in models:
    model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

dataset = pd.read_csv("data\clean\PREPROCESSED-NOTES.csv")

text_data = dataset["text"].to_list()

sdoh_columns = ["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment","behavior_alcohol", "behavior_tobacco","behavior_drug"]
sdoh_data = dataset[sdoh_columns].values.tolist()

X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size = .8)

y_train = np.array(y_train)
y_val = np.array(y_val)

max_seq_length = 100 #512

# Truncate and tokenize your input data
train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Convert labels to LongTensor
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

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

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainers = []
evaluation_results = []

training_args = TrainingArguments(
    output_dir=epoch_logs,
    logging_strategy='epoch',
    num_train_epochs=4,
    per_device_train_batch_size=8, #16  
    per_device_eval_batch_size=32, #64   
    warmup_steps=500,
    weight_decay=1e-5,
    logging_dir=tensor_logs,
    eval_steps=100,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=4
)

for i in range(len(models)): 
    trainer = Trainer(
        model=models[i],                 
        args=training_args,                  
        train_dataset=DataLoader(train_encodings, y_train[:, i]),         
        eval_dataset=DataLoader(val_encodings, y_val[:, i]),    
        compute_metrics=compute_metrics        
    )
    trainers.append(trainer)

    # Train the model
    trainer.train()

    # Evaluate the model
    result = trainer.evaluate()
    evaluation_results.append(result)

    # Save the model
    save_directory_i = f"saved_models/model_{i}" 
    os.makedirs(save_directory_i, exist_ok=True)
    models[i].save_pretrained(save_directory_i)
    tokenizer.save_pretrained(save_directory_i)

# evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)
