from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv("../data/PREPROCESSED-NOTES.csv")

text_data = dataset["text"].to_list()
sdoh_data = dataset["sdoh_community_present"].map(str).to_list()  # Convert labels to strings

X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size=.8, stratify=sdoh_data)

max_seq_length = 100  # or another appropriate length

class DataLoader(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        text = "classify: " + self.texts[idx]  # Prefix for classification
        label = self.labels[idx]

        # Encoding the input and target text
        encoding = tokenizer(text, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = tokenizer(label, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors='pt')

        return {"input_ids": encoding.input_ids.flatten(),
                "attention_mask": encoding.attention_mask.flatten(),
                "labels": target_encoding.input_ids.flatten()}

    def __len__(self):
        return len(self.texts)

train_dataset = DataLoader(X_train, y_train)
val_dataset = DataLoader(X_val, y_val)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Decode the predictions
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in predictions]

    # Decode the labels
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True).strip() for label in labels]

    # Convert decoded predictions and labels back to original label format if necessary
    # e.g., '0' -> 0, '1' -> 1, etc.
    # This step depends on how your labels are formatted.
    decoded_predictions = [int(pred) for pred in decoded_predictions]
    decoded_labels = [int(label) for label in decoded_labels]

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(decoded_labels, decoded_predictions, average='binary')
    acc = accuracy_score(decoded_labels, decoded_predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./t5result',
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

save_directory = "./new_saved_models"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)



