from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime


class Model:
    def __init__(self,  model,
                 epochs,
                 batch,
                 logs,
                 save_dir,
                 data) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.set_training_args(epochs, batch, logs)
        data.encode_data(self)
        self.set_trainer(
            train=data.train_dataset,
            val=data.val_dataset
        )
        self.save_dir = save_dir

    def set_training_args(self, epochs, batch, log_path) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_log = log_path + '/epoch/' + timestamp
        tensor_log = log_path + '/tensor/' + timestamp
        self.training_args = TrainingArguments(
            output_dir=epoch_log,
            logging_strategy='epoch',
            num_train_epochs=epochs,  # 4
            per_device_train_batch_size=batch, 
            per_device_eval_batch_size=batch,
            warmup_steps=500,
            weight_decay=1e-5, 
            logging_dir=tensor_log,
            eval_steps=100,
            evaluation_strategy="epoch"
        )

    def set_trainer(self, train, val):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=self.compute_metrics
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        eval_results = self.trainer.evaluate()
        print("Evaluation Results:", eval_results)

    def save_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)

    def compute_metrics(self,eval_pred):
        labels = eval_pred.label_ids
        preds = eval_pred.predictions.argmax(-1)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def run(self):
        self.train()
        self.evaluate()
        self.save_model()
