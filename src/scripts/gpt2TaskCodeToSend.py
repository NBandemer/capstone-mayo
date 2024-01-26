import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, TrainingArguments, Trainer
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split

# Define your scheduler
from transformers import get_linear_schedule_with_warmup


tokenizer = GPT2Tokenizer.from_pretrained("Locutusque/gpt2-large-medical")
model = GPT2ForSequenceClassification.from_pretrained("Locutusque/gpt2-large-medical")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

#dataset = pd.read_csv("/mnt/storage/manav_cap/data/clean/PREPROCESSED-NOTES.csv")
dataset = pd.read_csv("/Users/priyankat/Downloads/PREPROCESSED-NOTES.csv")
text_data = dataset["text"].to_list()[:1000]
sdoh_data = dataset["sdoh_community_present"].to_list()[:1000]

X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size = 0.8, stratify=sdoh_data)

max_seq_length = 100

# Truncate and tokenize your input data
train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

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

# Define evaluation metrics
def compute_metrics(pred): # may not be needed
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1
    }

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

total_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics= compute_metrics,
    optimizers=(optimizer, scheduler) #better generaliztion, minimizing loss
    #adding a scheduler to improve performance to our model
)

trainer.train()
trainer.evaluate()

save_directory = "./saved_models"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
