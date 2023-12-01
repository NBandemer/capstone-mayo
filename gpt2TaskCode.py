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

# Loading GPT-2 tokenizer and model for sequence classification
# Loading GPT-2 tokenizer and model for sequence classification
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Ensure the model is compatible with the tokenizer settings
configuration = GPT2ForSequenceClassification.config_class.from_pretrained("gpt2")
configuration.pad_token_id = tokenizer.pad_token_id
model = GPT2ForSequenceClassification(configuration)
#model = GPT2ForSequenceClassification.from_pretrained("gpt2")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GPT-2 uses the same token for end-of-sentence and padding.
#tokenizer.pad_token = tokenizer.eos_token

no_decay = ['bias', 'LayerNorm.weight'] #weight decy with a minor penalty during
optimizer_grouped_parameters = [ #no selects params added
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# Extracting text_data and sdoh_data from the dataset

#dataset = pd.read_csv("/mnt/storage/manav_cap/data/clean/PREPROCESSED-NOTES.csv")
dataset = pd.read_csv("/Users/priyankat/Downloads/PREPROCESSED-NOTES.csv")
text_data = dataset["text"].to_list()
sdoh_data = dataset["sdoh_community_present"].to_list()

X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size = 0.8, stratify=sdoh_data)

max_seq_length = 100

# Truncate and tokenize your input data for training and validation
#train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
#val_encodings = tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors='pt')

#custom Dataset class for loading training and validation data
class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert to tensor once here

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()  # Already a tensor, just clone and detach
        return item

    def __len__(self):
        return len(self.labels)

# Initialize the DataLoader for training and validation sets with the tokenized encodings
train_dataset = DataLoader(
    train_encodings,  # These should be the output from the tokenizer
    y_train           # These should be your labels, as a list or tensor
)

val_dataset = DataLoader(
    val_encodings,    # These should be the output from the tokenizer
    y_val             # These should be your labels, as a list or tensor
)

# Define evaluation metrics after training
def compute_metrics(pred): # may not be needed
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1
    }

#training args - need to adjust
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
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
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

save_directory = "./saved_models"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

evaluation_results = trainer.evaluate() # evaluation
print("Evaluation Results:", evaluation_results)
