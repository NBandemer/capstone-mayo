from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

dataset = pd.read_csv("data/PREPROCESSED-NOTES.csv")

text_data = dataset["text"].to_list()
sdoh_data = dataset["sdoh_community_present"].to_list()

X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size = .8, stratify=sdoh_data)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0, test_size = .01)

max_seq_length = 100 #512

# Truncate and tokenize your input data
train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

# print(X_train[0])
# print("Divider!!!!!!!!!!!!!!!!!!!!!!!!!!")
# first_row = {key: value[0] for key, value in train_encodings.items()}
# print(first_row)

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

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=7,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=1e-5,               
    logging_dir='./logs',            
    eval_steps=100                   
)

trainer = Trainer(
    model=model,                 
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
)

trainer.train()
trainer.evaluate()

# Saving & Loading the model<br>
save_directory = "/saved_models" 
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)