from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

import datetime
import os
import warnings

from helper import *

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

class TrainModel():
    def __init__(self, Sdoh_name, num_of_labels, model_name, epochs, batch, project_base_path):
        """
        Initialize the tokenizer and model for the class to use
        """
        # Suppress FutureWarning messages
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_of_labels)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.Sdoh_name = Sdoh_name
        self.num_of_labels = num_of_labels
        self.epochs = epochs
        self.batch = batch
        self.project_base_path = project_base_path

    def generate_model(self):
        # TODO: Clarify
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

        base_path_for_test_train_split = os.path.join(self.project_base_path, f"test_train_split/{self.Sdoh_name}/")

        # Reading the test_train_split data and converting it into lists for the tokenizer to use
        X_train = pd.read_csv(base_path_for_test_train_split + 'X_train.csv').iloc[:, 0].tolist()
        X_val = pd.read_csv(base_path_for_test_train_split + 'X_val.csv').iloc[:, 0].tolist()
        y_train = pd.read_csv(base_path_for_test_train_split + 'y_train.csv').iloc[:, 0].tolist()
        y_val = pd.read_csv(base_path_for_test_train_split + 'y_val.csv').iloc[:, 0].tolist()

        max_seq_length = 100 

        # Truncate and tokenize your input data
        train_encodings = self.tokenizer(X_train, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        val_encodings = self.tokenizer(X_val, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

        train_dataset = DataLoader(
            train_encodings,
            y_train
        )

        val_dataset = DataLoader(
            val_encodings,
            y_val
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        tensor_logs = os.path.join(self.project_base_path, f'logs/{self.Sdoh_name}/tensor_logs/logs_{timestamp}')
        os.makedirs(tensor_logs, exist_ok=True)

        epoch_logs = os.path.join(self.project_base_path, f'logs/{self.Sdoh_name}/epoch_logs/logs_{timestamp}')
        os.makedirs(epoch_logs, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=epoch_logs,
            logging_dir=tensor_logs,
            save_strategy='epoch',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch,  
            per_device_eval_batch_size=self.batch,
            weight_decay=1e-5,
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,                 
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=val_dataset,    
            compute_metrics=compute_metrics        
        )

        # trainer.train()
        # trainer.evaluate()

        # File path to store metrics (csv)
        metrics_base_path = os.path.join(self.project_base_path, f'metrics')
        os.makedirs(metrics_base_path, exist_ok=True)

        # convert to eval results to csv
        latest_checkpoint = get_latest_checkpoint(epoch_logs)

        json_path = os.path.join(latest_checkpoint, 'trainer_state.json')
        metric_save_path = f'{metrics_base_path}/{self.Sdoh_name}_eval_metric.csv'
        save_metrics_to_csv(json_path, f'{metrics_base_path}/{self.Sdoh_name}_eval_metric.csv')

        graph_path = os.path.join(self.project_base_path, f'graphs')
        os.makedirs(graph_path, exist_ok=True)

        plot_metrics_from_csv(metric_save_path, f'{graph_path}/{self.Sdoh_name}_metrics_plot.jpg')

        # plt.show(block=False)
        # plt.pause(10)
        # plt.close()

        # Saving the model
        save_directory = os.path.join(self.project_base_path, f'saved_models/{self.Sdoh_name}')
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        evaluation_results = trainer.evaluate()
        print("Evaluation Results:", evaluation_results)