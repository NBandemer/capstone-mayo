from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split

import datetime
import warnings
import shutil
import os

from helper import *

class MIMICDataset(Dataset):
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
    def __init__(self, *args, **kwargs):
        self.test = kwargs.pop('test', False)
        super().__init__(*args, **kwargs)

    def compute_metrics(self, eval_pred):
        """
        This function computes the metrics for the given model and inputs
        """
        compute_metrics(eval_pred, self.test)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This function computes the loss for the given model and inputs
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.weights if hasattr(self, 'weights') else None)
        return (loss, outputs) if return_outputs else loss

class Model():
    def __init__(self, Sdoh_name, num_of_labels, model_name, epochs, batch, project_base_path, balanced, weighted):
        """
        Initialize the tokenizer and model for the class to use
        """
        # Suppress FutureWarning messages
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_of_labels)

        # Initialize device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.Sdoh_name = Sdoh_name
        self.num_of_labels = num_of_labels
        self.epochs = epochs
        self.batch = batch
        self.project_base_path = project_base_path
        self.balanced = balanced
        self.weighted = weighted

    def train(self):
        data_path = os.path.join(self.project_base_path, f"data/test_train_split/{self.Sdoh_name}/")
        data_file = 'train.csv'
        df = pd.read_csv(os.path.join(data_path, data_file))
        df.dropna(subset=['text'], inplace=True)
        x = df['text']
        y = df[self.Sdoh_name]

        # Select 20% of training data for validation
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        
        if self.balanced and self.weighted:
            print("Both balanced and weighted flags are set, please set only one of them")
            exit(1)

        # Handle class imbalance in training data
        if self.balanced:
            df_train = pd.DataFrame({'X': X_train, 'y': y_train})
            balanced_train = balance_data(df_train)

            # Convert back to lists if needed
            list_train_x = balanced_train['X'].tolist()
            list_train_y = balanced_train['y'].tolist()
            list_val_x = X_val.tolist()
            list_val_y = y_val.tolist()
        else:
            list_train_x = X_train.tolist()
            list_train_y = y_train.tolist()
            list_val_x = X_val.tolist()
            list_val_y = y_val.tolist()

        max_seq_length = 128 

        # Truncate and tokenize your input data
        train_encodings = self.tokenizer(list_train_x, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        val_encodings = self.tokenizer(list_val_x, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        
        train_dataset = MIMICDataset(
            train_encodings,
            list_train_y
        )

        val_dataset = MIMICDataset(
            val_encodings,
            list_val_y
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        tensor_logs = os.path.join(self.project_base_path, f'logs/{self.Sdoh_name}/tensor_logs/logs_{timestamp}')
        os.makedirs(tensor_logs, exist_ok=True)

        epoch_logs = os.path.join(self.project_base_path, f'logs/{self.Sdoh_name}/epoch_logs/logs_{timestamp}')
        os.makedirs(epoch_logs, exist_ok=True)

        early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

        # train_loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=self.batch, shuffle=True)
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
        
        len(train_dataset) // self.batch

        num_training_steps = (len(train_dataset) // self.batch) * self.epochs
        num_warmup_steps = (len(train_dataset) // self.batch) * 0.1

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[early_stopping],
            test=False,
            optimizers=(optimizer, get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps))
        )

        if self.weighted:
            self.weights = get_class_weights(list_train_y)

        trainer.train()

        # for epoch in range(3):
        #     for batch in train_loader:
        #         optim.zero_grad()
        #         input_ids = batch['input_ids'].to(self.device)
        #         attention_mask = batch['attention_mask'].to(self.device)
        #         labels = batch['labels'].to(self.device)
        #         outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        #         loss = outputs[0]
        #         loss.backward()
        #         optim.step()

        print("Training complete")

        graph_path = os.path.join(self.project_base_path, f'graphs')
        os.makedirs(graph_path, exist_ok=True)

        plot_metric_from_tensor(tensor_logs, f'{graph_path}/{self.Sdoh_name}_metrics_plot.jpg')

        # Saving the model
        save_dir = f'saved_models/{self.Sdoh_name}'
        save_dir += '_balanced' if self.balanced else '_weighted' if self.weighted else ''

        save_directory = os.path.join(self.project_base_path, save_dir)
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def test(self):
        data_path = os.path.join(self.project_base_path, f"data/test_train_split/{self.Sdoh_name}")
        
        data_file = 'test.csv'
        df = pd.read_csv(os.path.join(data_path, data_file))

        df.dropna(subset=['text'], inplace=True)
        eval_inputs = df['text'].tolist()
        eval_labels = df[self.Sdoh_name].tolist()
        
        max_seq_length = 128

        eval_encodings = self.tokenizer(eval_inputs, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

        eval_dataset = MIMICDataset(
            eval_encodings,
            eval_labels
        )

        saved_model = os.path.join(self.project_base_path, f'saved_models/{self.Sdoh_name}')
        model =  AutoModelForSequenceClassification.from_pretrained(saved_model)

        trainer = CustomTrainer(
            model=model,
            test=True,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        set_sdoh(self.Sdoh_name)
        results = trainer.evaluate()

        results.cm.plot()
        plt.show()

        results.roc.plot()
        plt.show()

        print("Classification Report:", results.classification_report)
        # print("Evaluation Results:", results)

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