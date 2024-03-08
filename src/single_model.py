import os
import random
import shutil
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from transformers import EarlyStoppingCallback

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

from helper import plot_metric_from_tensor

BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 50
pd.options.display.max_columns = None

# SDOHs
sdohs = ["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"]

sdoh_dict = {
    "sdoh_community_present": 2,
    "sdoh_community_absent": 2,
    "sdoh_education": 2,
    "sdoh_economics": 3,
    "sdoh_environment": 3,
    "behavior_alcohol": 5,
    "behavior_tobacco": 5,
    "behavior_drug": 5
}
# SBDH labels
# Community
SDOH_COMMUNITY_PRESENT_LABELS, SDOH_COMMUNITY_ABSENT_LABELS, SDOH_EDUCATION_LABELS = [0, 1], [0,1], [0,1]

# Econ and Env
SDOH_ECONOMICS_LABELS, SDOH_ENVIRONMENT_LABELS = [0, 1, 2], [0, 1, 2]

# Subtance
BEHAVIOR_ALCOHOL_LABELS, BEHAVIOR_TOBACCO_LABELS, BEHAVIOR_DRUG_LABELS = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]

# SBDH Indices
# Community
SDOH_COMMUNITY_PRESENT_INDICES = range(0, 2)
SDOH_COMMUNITY_ABSENT_INDICES = range(2, 4)

# Social
SDOH_EDUCATION_INDICES = range(4, 6)
SDOH_ECONOMICS_INDICES = range(6, 9)
SDOH_ENVIRONMENT_INDICES = range(9, 12)

# Subtance
BEHAVIOR_ALCOHOL_INDICES = range(12, 17)
BEHAVIOR_TOBACCO_INDICES = range(17, 22)
BEHAVIOR_DRUG_INDICES = range(22, 27)
MAX_LENGTH = 128

ALL_LABELS = SDOH_COMMUNITY_PRESENT_LABELS + SDOH_COMMUNITY_ABSENT_LABELS + SDOH_EDUCATION_LABELS + SDOH_ECONOMICS_LABELS + SDOH_ENVIRONMENT_LABELS + BEHAVIOR_ALCOHOL_LABELS + BEHAVIOR_TOBACCO_LABELS + BEHAVIOR_DRUG_LABELS

id2label = {k:l for k, l in enumerate(ALL_LABELS)}
label2id = {l:k for k, l in enumerate(ALL_LABELS)}


base_path = Path(__file__).parent.parent.resolve()

def preprocess_function(row: pd.Series, tokenizer):
    labels = row.iloc[1:]
    row = tokenizer(row["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    row["label"] = labels
    return row



"""
Test-train split
"""

def test_train_split():
    """
    This function creates the test_train_split for all the SDoH from the pre processed data
    """
    df = pd.read_csv(os.path.join(base_path, "data/PREPROCESSED-NOTES-NEW.csv"), index_col=0)
    df = df.dropna(subset=["text"])

    X = df["text"]
    y = df.iloc[:, 1:9]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

    # Combine features and target variables for train and test sets
    train_data = pd.concat([X_train, y_train], axis=1).dropna()
    test_data = pd.concat([X_test, y_test], axis=1)

    # Create folder for each category
    category_data_path = os.path.join(base_path, "data/test_train_split")
    os.makedirs(category_data_path, exist_ok=True)

    # Save train and test data to seprate csvs
    train_data.to_csv(f"{category_data_path}/train.csv", index=False)
    test_data.to_csv(f"{category_data_path}/test.csv", index=False)


# test_train_split()

def get_class_weights(y_train):
    """
    This function calculates the weighted loss for the given training data
    """
    y_train = y_train.iloc[:, 1:]
    all_weights = []

    for sdoh in sdohs:
        sdoh_y = y_train[sdoh]
        class_weights = compute_class_weight('balanced', classes=np.unique(sdoh_y), y=sdoh_y)
        # Create a dictionary to hold the class weights
        weight_dict = {i: float(class_weights[i]) for i in range(len(class_weights))}
        # Convert the dictionary to a tensor
        weights = torch.tensor([weight_dict[key] for key in range(len(weight_dict))])
        all_weights.append(weights)
    # Assuming your list of tensors is named 'tensor_list'
    new_all_weights = [float(item) for sublist in all_weights for item in sublist]

    return new_all_weights

""" 
Load the split data
"""
# train = pd.read_csv(os.path.join(base_path, "data/test_train_split/train.csv"))

# weights = torch.tensor(get_class_weights(train))


# train = pd.get_dummies(train, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"], dtype=int)

# X = train['text']
# y = train.iloc[:, 1:]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

# train_data = pd.concat([X_train, y_train], axis=1)
# test_data = pd.concat([X_test, y_test], axis=1)

# train_data = train_data.apply(preprocess_function(tokenizer=A), axis=1)
# test_data = test_data.apply(preprocess_function, axis=1)

# test = pd.read_csv("/home/nano/Code/ML/Mayo/data/test_train_split/test.csv")
# train = pd.get_dummies(train, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"])
# test = pd.get_dummies(test, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"])
# train = train.apply(preprocess_function, axis=1)
# test = test.apply(preprocess_function, axis=1)
"""
Metrics
- Get predictions from logits
- Compute metrics
"""

def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    
    index = 0

    for sdoh in sdohs:
        sdoh_indices = globals()[sdoh.upper() + "_INDICES"]
        sdoh_length = len(sdoh_indices)
        best_class_index = np.argmax(logits[:, sdoh_indices], axis=-1)
        ret[list(range(len(ret))), index + best_class_index] = 1
        index += sdoh_length
    
    return ret

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    final_metrics = {}
    
    # Deduce predictions from logits
    predictions = get_preds_from_logits(logits)
    
    for sdoh in sdohs:
        index = globals()[sdoh.upper() + "_INDICES"]

        final_metrics[f'f1_micro_{sdoh}'] = f1_score(labels[:, index], predictions[:, index], average="micro")
        final_metrics[f'f1_macro_{sdoh}'] = f1_score(labels[:, index], predictions[:, index], average="macro")

        # final_metrics[f'accuracy_micro_{sdoh}'] = accuracy_score(labels[:, index], predictions[:, index], average="micro")
        final_metrics[f'accuracy_macro_{sdoh}'] = accuracy_score(labels[:, index], predictions[:, index])

        print(f"Classification report for {sdoh}: ")
        print(classification_report(labels[:, index], predictions[:, index], zero_division=0))

    # The global f1_metrics
    final_metrics["f1_micro"] = f1_score(labels, predictions, average="micro")
    final_metrics["f1_macro"] = f1_score(labels, predictions, average="macro")

    # final_metrics["accuracy_micro"] = accuracy_score(labels, predictions, average="micro")
    final_metrics["accuracy_macro"] = accuracy_score(labels, predictions)
    
    return final_metrics

"""
Trainer
"""

class MultiTaskClassificationTrainer(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights
    
    """
    Overriding the compute_loss method
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        losses = []

        self.group_weights = self.group_weights.to(logits.device) 
        
        for sdoh in sdohs:
            sdoh_indices = globals()[sdoh.upper() + "_INDICES"]
            sdoh_slice = slice(sdoh_indices.start, sdoh_indices.stop)

            losses.append(torch.nn.functional.cross_entropy(logits[:, sdoh_slice], labels[:, sdoh_slice], weight=self.group_weights[sdoh_slice]))
        loss = sum(losses)
        return (loss, outputs) if return_outputs else loss
    
"""
Set training parameters
"""

class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")



"""
Train the model
- Should make this manual but not necessary
"""
def train():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)
    model.to(torch.device("cuda"))

    optimizer = AdamW(model.parameters(),
                lr=LEARNING_RATE, 
                eps=1e-8)

    train = pd.read_csv(os.path.join(base_path, "data/test_train_split/train.csv"))

    weights = torch.tensor(get_class_weights(train))

    train = pd.get_dummies(train, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"], dtype=int)

    X = train['text']
    y = train.iloc[:, 1:]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data = train_data.apply(preprocess_function, tokenizer=tokenizer, axis=1)
    test_data = test_data.apply(preprocess_function, tokenizer=tokenizer, axis=1)

    scheduler = get_linear_schedule_with_warmup(num_warmup_steps=(len(train_data) // BATCH_SIZE) * 0.1, num_training_steps=(len(train_data) // BATCH_SIZE) * EPOCHS, optimizer=optimizer)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

    training_args = TrainingArguments(
        output_dir="./logs/output-dir",
        logging_dir="./logs/tensor_logs",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="f1_macro",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

    trainer = MultiTaskClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data.values,
        eval_dataset=test_data.values,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback, early_stopping],
        optimizers=(optimizer, scheduler),
        group_weights=weights
    )

    trainer.train()

    print("Training complete")

    graph_path = os.path.join(base_path, f'graphs')
    os.makedirs(graph_path, exist_ok=True)

    plot_metric_from_tensor("./logs/tensor_logs", f'{graph_path}/metrics_plot.jpg')

    # Saving the model
    save_directory = os.path.join(base_path, f'saved_models')
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

train()

def test():
    test = pd.read_csv(os.path.join(base_path, "data/test_train_split/test.csv"))

    test = pd.get_dummies(test, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"], dtype=int)

    X = test['text']
    y = test.iloc[:, 1:]

    test_data = pd.concat([X, y_train], axis=1)
    test_data = test_data.apply(preprocess_function, axis=1)
       
    saved_model = '/home/nano/Code/ML/Mayo/models/camembert-fine-tuned/checkpoint-3525'
    model =  AutoModelForSequenceClassification.from_pretrained(saved_model)

    trainer = MultiTaskClassificationTrainer(
        model=model,
        eval_dataset=test_data.values,
        compute_metrics=compute_metrics,
    )

    model.eval()
    results = trainer.evaluate()
    print("Evaluation Results:", results)

# test()
    
    # # Save evaluation results to a CSV file
    # results_df = pd.DataFrame([results])
    # results_path = os.path.join(project_base_path, f'test_results/{self.Sdoh_name}_before')
    # os.makedirs(results_path, exist_ok=True)
    # results_df.to_csv(f"{results_path}/results.csv", index=False)
    # print("Evaluation results saved to:", results_path)

    # # Clean up temp files
    # tmp_dir = os.path.join(os.getcwd(), 'tmp_trainer')
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)

# seed_val = 17

# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# print(device)
# for epoch in tqdm(range(1, EPOCHS+1)):
#     model.train()

#     loss_train_total = 0

#     progress_bar = tqdm(train_data, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
#     for batch in progress_bar:

#         model.zero_grad()
    
#         batch = tuple(b.to(device) for b in batch)
    
#         inputs = {'input_ids':      batch[0],
#             'attention_mask': batch[1],
#             'labels':         batch[2],
#             }       

#         outputs = model(**inputs)
    
#         loss = outputs[0]
#         loss_train_total += loss.item()
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()
#         scheduler.step()
    
#         progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
    
    
#     torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
    
#     tqdm.write(f'\nEpoch {epoch}')

#     loss_train_avg = loss_train_total/len(dataloader_train)            
#     tqdm.write(f'Training loss: {loss_train_avg}')

#     val_loss, predictions, true_vals = evaluate(model, dataloader_validation, device)
#     val_f1 = f1_score_func(predictions, true_vals)
#     tqdm.write(f'Validation loss: {val_loss}')
#     tqdm.write(f'F1 Score (Weighted): {val_f1}')

# _, predictions, true_vals = evaluate(model, dataloader_validation, device)