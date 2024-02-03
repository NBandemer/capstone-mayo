import datetime
import os
import torch
from torch.optim import AdamW  # variant of Adam with weight decay
from torch.utils.data import Dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, TrainingArguments, Trainer
import pandas as pd
import json
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

# Initialize tokenizer, this is standard approach with GPT-2
# Loading GPT-2 tokenizer and model for sequence classification
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 uses the same token for end-of-sentence and padding.
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Ensure the model is compatible with the tokenizer settings
configuration = GPT2ForSequenceClassification.config_class.from_pretrained("gpt2")
configuration.pad_token_id = tokenizer.pad_token_id
model = GPT2ForSequenceClassification(configuration)
# model = GPT2ForSequenceClassification.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Set up no decay for certain model parameters to avoid regularization on them
no_decay = ['bias', 'LayerNorm.weight']  # weight decay with a minor penalty during
optimizer_grouped_parameters = [  # no selects params added
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
dataset = pd.read_csv("/Users/priyankat/PycharmProjects/Mayo/PREPROCESSED-NOTES.csv")

text_data = dataset["text"].to_list()
sdoh_data = dataset["behavior_alcohol"].to_list()

timestamp_fortrain = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(text_data, sdoh_data, random_state=0, train_size=0.8,
                                                  stratify=sdoh_data)
max_seq_length = 100  # actually 50 but increase to accomadate outliers

# Calculate the number of trainable parameters in the model
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_MB = num_trainable_params * 4 / (1024 ** 2)
effective_batch = 8 / (50*4*model_size_MB) #gpu/seqlength * 4 * model size

train_encodings = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

# custom Dataset class for loading training and validation data
class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        """ :rtype: object """
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)  # Converting to tensor , maybe use just 'labels'

    def __getitem__(self, idx):
        try:
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx].clone().detach()  # Already a tensor, just clone and detach
            return item
        except Exception as e:
            print(f"index error: {idx}: {e}")
            return None

    def __len__(self):
        return len(self.labels) # detach from tensor device

# Initialize the DataLoader for training and validation sets with the tokenized encodings
train_dataset: DataLoader = DataLoader(
    train_encodings,  # These should be the output from the tokenizer
    y_train  # These should be your labels, as a list or tensor
)

val_dataset = DataLoader(
    val_encodings,  # These should be the output from the tokenizer
    y_val  # These should be your labels, as a list or tensor
)

tensor_logs = f'./logs/tensor_logs/{timestamp_fortrain}' #create seperate logs for tensor/epoch
os.makedirs(tensor_logs, exist_ok=True)
epoch_logs = f'./logs/epoch_logs/{timestamp_fortrain}'
os.makedirs(epoch_logs, exist_ok=True)

# training args - need to adjust
training_args = TrainingArguments(
    output_dir= epoch_logs,  # change to epoch log directory, convert to a text
    logging_strategy='epoch',  # characterize as epoch
    num_train_epochs=4,
    per_device_train_batch_size=2,  # cpu constraint,  64 approp
    per_device_eval_batch_size=2,  # gradient accum if batch size of two, 64 approp
    save_strategy= 'epoch',
    warmup_steps=500,
    weight_decay=1e-5,
    logging_dir= tensor_logs,  # change to tensor logs
    eval_steps=100,
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
evaluation_results = trainer.evaluate()

#readable results
latest_checkpoint = get_latest_checkpoint(epoch_logs) # latest checkpoint update to csv
json_path = os.path.join(latest_checkpoint, 'trainer_state.json')
save_metrics_to_csv(json_path, 'eval_metric.csv') #update metrics
plot_metrics_from_csv('eval_metric.csv', 'graphs')

save_directory = "saved_models/gpt2"

os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Evaluation Results:", evaluation_results)






