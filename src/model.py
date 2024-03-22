import random
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from transformers import TrainingArguments, Trainer
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import EarlyStoppingCallback

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

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
        self.cv = kwargs.pop('cv', False)
        super().__init__(*args, **kwargs)

    def compute_metrics(self, eval_pred):
        """
        This function computes the metrics for the given model and inputs
        """
        compute_metrics(eval_pred, self.cv, self.test)

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
    def __init__(self, Sdoh_name, num_of_labels, model_name, epochs, batch, project_base_path, balanced, weighted, output_dir=None, cv=None):
        """
        Initialize the tokenizer and model for the class to use
        """
        # Suppress FutureWarning messages
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

        # Initialize device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model_name = model_name
        self.Sdoh_name = Sdoh_name
        self.num_of_labels = num_of_labels
        self.epochs = epochs
        self.batch = batch
        self.project_base_path = project_base_path
        self.balanced = balanced
        self.weighted = weighted
        self.output_dir = output_dir
        self.cv = cv

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    def get_training_data(self):
        """
        This function gets the training data for the given SDoH
        """
        data_path = os.path.join(self.project_base_path, f"data/test_train_split/{self.Sdoh_name}/")
        data_file = 'train.csv'
        df = pd.read_csv(os.path.join(data_path, data_file))
        df.dropna(subset=['text'], inplace=True)
        x = df['text']
        y = df[self.Sdoh_name]

        return x, y

    def train(self):
        if self.balanced and self.weighted:
            print("Both balanced and weighted flags are set, please set only one of them")
            exit(1)

        x,y = self.get_training_data()

        # Training constants
        MAX_LENGTH = 128 
        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
        current_fold = 1

        if self.weighted:
            self.weights = get_class_weights(y)

        if self.cv:
            # Implement 5-fold stratified cross val
            skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            split_iterator = skf.split(x, y)
        
        else:
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
            max_index = len(x) - 1  # Get the maximum valid index of the dataframe
            split_iterator = [(list(X_train.index[X_train.index <= max_index]), list(X_val.index[X_val.index <= max_index]))]

        cross_val_accuracies = []
        cross_val_f1s = []
        cross_val_aucs = []
        cross_val_losses = []

        for train, test in split_iterator:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_of_labels)
            model.to(self.device)
            optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

            X_train, X_val = x.iloc[train], x.iloc[test]
            y_train, y_val = y.iloc[train], y.iloc[test]

            epoch_training_steps = len(X_train) // self.batch
            num_training_steps = epoch_training_steps * self.epochs
            num_warmup_steps = epoch_training_steps * 0.1
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            optimizers = (optimizer, scheduler)

            # Handle class imbalance in training data
            if self.balanced:
                df_train = pd.DataFrame({'X': X_train, 'y': y_train})
                balanced_train = balance_data(df_train)

                # Convert back to lists if needed
                list_train_x = balanced_train['X'].tolist()
                list_train_y = balanced_train['y'].tolist()
            else:
                list_train_x = X_train.tolist()
                list_train_y = y_train.tolist()

            list_val_x = X_val.tolist()
            list_val_y = y_val.tolist()

            # Truncate and tokenize your input data
            train_encodings = self.tokenizer(list_train_x, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
            val_encodings = self.tokenizer(list_val_x, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
            
            train_dataset = MIMICDataset(
                train_encodings,
                list_train_y
            )

            val_dataset = MIMICDataset(
                val_encodings,
                list_val_y
            )

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            log_dir = self.project_base_path if self.output_dir is None else self.output_dir

            tensor_logs = os.path.join(log_dir, f'logs/{self.Sdoh_name}/tensor_logs/logs_{timestamp}')
            os.makedirs(tensor_logs, exist_ok=True)

            epoch_logs = os.path.join(log_dir, f'logs/{self.Sdoh_name}/epoch_logs/logs_{timestamp}')
            os.makedirs(epoch_logs, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=epoch_logs,
                logging_dir=tensor_logs,
                save_strategy='epoch',
                logging_strategy='epoch',
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch,  
                per_device_eval_batch_size=self.batch,
                evaluation_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model='eval_f1'
            )
        
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[early_stopping],
                compute_metrics=compute_metrics_train,
                test=False,
                cv=self.cv,
                optimizers=optimizers,
            )

            print(f'Starting Training{" Fold " + str(current_fold) if self.cv else ""}')

            trainer.train()

            print(f'Finished Training{" Fold " + str(current_fold) if self.cv else ""}')

            # If cross val, run evaluation on the model

            if self.cv:
                results = trainer.evaluate()
                accuracy = results.get('eval_accuracy')
                f1 = results.get('eval_f1')
                auc = results.get('eval_auc')
                loss = results.get('eval_loss')
                cross_val_accuracies.append(accuracy)
                cross_val_f1s.append(f1)
                cross_val_aucs.append(auc)
                cross_val_losses.append(loss)

            
            graph_dir = os.path.join(self.project_base_path, f'graphs/')
            save_dir = os.path.join(self.project_base_path, f'saved_models/')

            # Configure directory paths depending on config
            if self.balanced:
                graph_dir += 'balanced/'
                save_dir += 'balanced/'
            elif self.weighted:
                graph_dir += 'weighted/'
                save_dir += 'weighted/'
            else:
                graph_dir += 'standard/'
                save_dir += 'standard/'

            graph_dir += self.Sdoh_name
            save_dir += self.Sdoh_name

            # Create directories
            os.makedirs(graph_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            # Cross validation is not meant to generate a final model, its just for metrics so dont save
            if not self.cv:
                plot_metric_from_tensor(tensor_logs, f'{graph_dir}/')
                model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)

            current_fold += 1

        if self.cv:
            df_data = {
                'accuracy': cross_val_accuracies,
                'f1': cross_val_f1s,
                'auc': cross_val_aucs,
                'loss': cross_val_losses
            }

            cv_path = os.path.join(self.project_base_path, f'test_results/cv/')
            if self.balanced:
                cv_path += 'balanced/'
            elif self.weighted:
                cv_path += 'weighted/'
            os.makedirs(cv_path, exist_ok=True)
            
            df = pd.DataFrame(df_data)
            df.to_csv(f'{cv_path}{self.Sdoh_name}.csv', index=False)

            # print(f'Cross Validation Results for {self.Sdoh_name}')
            # print(f'Accuracies: {cross_val_accuracies}')
            # print(f'F1 Scores: {cross_val_f1s}')           
            # print(f'AUCs: {cross_val_aucs}')  
            # print(f'Losses: {cross_val_losses}')              
            # print(f'Average Accuracy: {np.mean(cross_val_accuracies)}')
            # print(f'Average F1: {np.mean(cross_val_f1s)}')
            # print(f'Average AUC: {np.mean(cross_val_aucs)}')
            # print(f'Average Loss: {np.mean(cross_val_losses)}')

    def test(self):
        if self.cv:
            print("Cross validation is not supported in test mode")
            exit(1)

        set_helper_sdoh(self.Sdoh_name)
        
        data_path = os.path.join(self.project_base_path, f"data/test_train_split/{self.Sdoh_name}")
        
        data_file = 'test.csv'
        test_df = pd.read_csv(os.path.join(data_path, data_file))

        test_df.dropna(subset=['text'], inplace=True)
        test_inputs = test_df['text'].tolist()
        test_labels = test_df[self.Sdoh_name].tolist()
        
        max_seq_length = 128

        test_encodings = self.tokenizer(test_inputs, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

        test_dataset = MIMICDataset(
            test_encodings,
            test_labels
        )

        saved_models_dir = os.path.join(self.project_base_path, f'saved_models/')
        sdoh_dir = ''

        if self.balanced:
            sdoh_dir += 'balanced/'
        elif self.weighted:
            sdoh_dir += 'weighted/'
        else:
            sdoh_dir += 'standard/'
            
        sdoh_dir += self.Sdoh_name
        results_dir = os.path.join(self.project_base_path, f'test_results/{sdoh_dir}')
        os.makedirs(results_dir, exist_ok=True)
        
        model =  AutoModelForSequenceClassification.from_pretrained(os.path.join(saved_models_dir, sdoh_dir))
        
        eval_trainer = CustomTrainer(
            model=model,
            test=True,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        results = eval_trainer.evaluate()

        cm = results.get('eval_cm')

        cm.plot()
        plt.savefig(f"{results_dir}/confusion_matrix.jpg")
        plt.close()
        
        curves = results.get('eval_roc')
        roc_dir = os.path.join(results_dir, 'roc')
        os.makedirs(roc_dir, exist_ok=True)

        # for display, best_threshold in curves:
        #     # Plot the ROC curve
        #     display.plot()
            
        #     # Set titles and labels
        #     plt.title(f'ROC Curve for {display.estimator_name}')
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
            
        #     # Add annotation for the best threshold
        #     plt.text(0.5, 0.5, f'Best Threshold: {best_threshold:.4f}', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
        #     # Save the figure as JPG with the estimator name
        #     plt.savefig(f'{roc_dir}/{display.estimator_name}.jpg')

        plot_roc(curves, roc_dir, self.Sdoh_name)

        report_df = results.get('eval_classification_report')
        report_df.to_csv(f"{results_dir}/classification_report.csv")

         # Save evaluation results to a CSV file
        results_df = pd.DataFrame([results]).drop(columns=['eval_cm', 'eval_roc', 'eval_classification_report'])
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(f"{results_dir}/results.csv", index=False)
        print("Evaluation results saved to:", results_dir)

        # Clean up temp files
        tmp_dir = os.path.join(os.getcwd(), 'tmp_trainer')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        # plt.show()