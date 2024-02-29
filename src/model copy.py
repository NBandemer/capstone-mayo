import numpy as np
from sklearn.utils import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback

import torch
from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

import datetime
import warnings
import shutil
import os

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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overriding the compute_loss function to apply weighted loss function
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def get_class_weights(self, y_train):
        """
        This function calculates the weighted loss for the given training data
        """
        # Calculate the class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        # Create a dictionary to hold the class weights
        weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        # Convert the dictionary to a tensor
        weights = torch.tensor([weight_dict[key] for key in range(len(weight_dict))])
        print("Class Weights:", weights)
        self.weights = weights

class Model():
    def __init__(self, Sdoh_name, num_of_labels, model_name, epochs, batch, project_base_path):
        """
        Initialize the tokenizer and model for the class to use
        """
        # Suppress FutureWarning messages
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_of_labels)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.Sdoh_name = Sdoh_name
        self.num_of_labels = num_of_labels
        self.epochs = epochs
        self.batch = batch
        self.project_base_path = project_base_path

    def load_data(self):
        data_path = os.path.join(self.project_base_path, f"data/test_train_split/{self.Sdoh_name}/")
        data_file = 'train.csv'
        df = pd.read_csv(os.path.join(data_path, data_file))

        x = df['text']
        y = df[self.Sdoh_name]

        # Select 20% of training data for validation
        return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    def balance_data(self, X_train, y_train):
        # Handle class imbalance in training data
        df_train = pd.DataFrame({'X': X_train, 'y': y_train})
        balanced_train = balance_data(df_train)
        # Convert back to lists if needed
        list_train_x = balanced_train['X'].tolist()
        list_train_y = balanced_train['y'].tolist()
        return list_train_x,list_train_y

    def train(self):
        # X_train, X_val, y_train, y_val = self.load_data()
        # # X_train, y_train = self.balance_data(X_train, y_train)
        data_path = os.path.join(self.project_base_path, f"data/test_train_split/{self.Sdoh_name}/")
        data_file = 'train.csv'
        df = pd.read_csv(os.path.join(data_path, data_file))

        x = df['text']
        y = df[self.Sdoh_name]

        # Select 20% of training data for validation
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle class imbalance in training data
        df_train = pd.DataFrame({'X': X_train, 'y': y_train})
        balanced_train = balance_data(df_train)

        # Convert back to lists if needed
        # list_train_x = balanced_train['X'].tolist()
        # list_train_y = balanced_train['y'].tolist()
        # list_val_x = X_val.tolist()
        # list_val_y = y_val.tolist()
    
        max_seq_length = 128 

        # Truncate and tokenize your input data
        train_encodings = self.tokenizer(X_train.tolist(), truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        val_encodings = self.tokenizer(X_val.tolist(), truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        
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
        # os.makedirs(tensor_logs, exist_ok=True)

        epoch_logs = os.path.join(self.project_base_path, f'logs/{self.Sdoh_name}/epoch_logs/logs_{timestamp}')
        # os.makedirs(epoch_logs, exist_ok=True)

        early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        training_args = TrainingArguments(
            output_dir=epoch_logs,
            logging_dir=tensor_logs,
            save_strategy='epoch',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch,  
            per_device_eval_batch_size=self.batch,
            weight_decay=1e-5,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
            optimizers=(optimizer, get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(len(train_dataset) // self.batch) * 0.1, num_training_steps=(len(train_dataset) // self.batch) * self.epochs))
        )

        # trainer.get_class_weights(y_train)
        trainer.train()
        print("Training complete")

        graph_path = os.path.join(self.project_base_path, f'graphs')
        os.makedirs(graph_path, exist_ok=True)

        plot_metric_from_tensor(tensor_logs, f'{graph_path}/{self.Sdoh_name}_metrics_plot.jpg')

        # Saving the model
        save_directory = os.path.join(self.project_base_path, f'saved_models/{self.Sdoh_name}')
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def test(self):
        data_path = os.path.join(self.project_base_path, f"test_train_split/{self.Sdoh_name}/")
        x_data = 'X_val.csv'
        y_data = 'y_val.csv' 

        #Load in the evaluation data sets
        x_eval = pd.read_csv(os.path.join(data_path, x_data)).iloc[:, 0].tolist()
        y_eval = pd.read_csv(os.path.join(data_path, y_data)).iloc[:, 0].tolist()
        
        max_seq_length = 128
        x_encodings = self.tokenizer(x_eval, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

        eval_dataset = DataLoader(
            x_encodings,
            y_eval
        )

        saved_model = os.path.join(self.project_base_path, f'saved_models/{self.Sdoh_name}')
        model =  AutoModelForSequenceClassification.from_pretrained(saved_model)

        trainer = Trainer(
            model=model,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        set_sdoh(self.Sdoh_name)
        results = trainer.evaluate()
        print("Evaluation Results:", results)

         # Save evaluation results to a CSV file
        results_df = pd.DataFrame([results])
        results_path = os.path.join(self.project_base_path, f'test_results/{self.Sdoh_name}')
        os.makedirs(results_path, exist_ok=True)
        results_df.to_csv(f"{results_path}/results.csv", index=False)
        print("Evaluation results saved to:", results_path)

        # Clean up temp files
        tmp_dir = os.path.join(os.getcwd(), 'tmp_trainer')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)