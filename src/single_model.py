import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 50

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

ALL_LABELS = SDOH_COMMUNITY_PRESENT_LABELS + SDOH_COMMUNITY_ABSENT_LABELS + SDOH_EDUCATION_LABELS + SDOH_ECONOMICS_LABELS + SDOH_ENVIRONMENT_LABELS + BEHAVIOR_ALCOHOL_LABELS + BEHAVIOR_TOBACCO_LABELS + BEHAVIOR_DRUG_LABELS

id2label = {k:l for k, l in enumerate(ALL_LABELS)}
label2id = {l:k for k, l in enumerate(ALL_LABELS)}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)


def preprocess_function(row: pd.Series):
    labels = [0] * len(ALL_LABELS)

    for key, value in sdoh_dict.items():
        for index in range(value):
            if row[f"{key}_{index}"] == 1:
                start_index = globals()[key.upper() + "_INDICES"].start
                labels[start_index + index] = 1
    # row_sdohs = row[2:10]
    # labels = row[2:len(row) - 2].values
    # for key, val in enumerate(labels):
    #     print(key,val)
    # print(labels)
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
    df = pd.read_csv("/home/nano/Code/ML/Mayo/data/PREPROCESSED-NOTES-NEW.csv", index_col=0)
    df = df.dropna(subset=["text"])

    dataset = df

    X = dataset["text"]
    y = dataset.iloc[:, 1:9]
    print(y.head())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

    # Combine features and target variables for train and test sets
    train_data = pd.concat([X_train, y_train], axis=1).dropna()
    test_data = pd.concat([X_test, y_test], axis=1)

    # Create folder for each category
    category_data_path = "/home/nano/Code/ML/Mayo/data/test_train_split"
    os.makedirs(category_data_path, exist_ok=True)

    # Save train and test data to seprate csvs
    train_data.to_csv(f"{category_data_path}/train.csv", index=False)
    test_data.to_csv(f"{category_data_path}/test.csv", index=False)


# test_train_split()

"""
Load the split data
"""
train = pd.read_csv("/home/nano/Code/ML/Mayo/data/test_train_split/train.csv")
test = pd.read_csv("/home/nano/Code/ML/Mayo/data/test_train_split/test.csv")
train = pd.get_dummies(train, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"])
test = pd.get_dummies(test, columns=["sdoh_community_present", "sdoh_community_absent", "sdoh_education", "sdoh_economics", "sdoh_environment", "behavior_alcohol", "behavior_tobacco", "behavior_drug"])
train = train.apply(preprocess_function, axis=1)
test = test.apply(preprocess_function, axis=1)
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
        print('best',best_class_index)
        index += sdoh_length
    
    print(ret)
    return ret

example = np.random.uniform(-2, 2, (1, 27))
print(get_preds_from_logits(example))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    final_metrics = {}
    
    # Deduce predictions from logits
    predictions = get_preds_from_logits(logits)
    
    # Get f1 metrics for global scoring. Notice that f1_micro = accuracy
    final_metrics["f1_micro_for_global_score"] = f1_score(labels[:, GLOBAL_SCORE_INDICES], predictions[:, GLOBAL_SCORE_INDICES], average="micro")
    final_metrics["f1_macro_for_global_score"] = f1_score(labels[:, GLOBAL_SCORE_INDICES], predictions[:, GLOBAL_SCORE_INDICES], average="macro")
    
    # Get f1 metrics for causes
    final_metrics["f1_micro_for_causes"] = f1_score(labels[:, CAUSE_INDICES], predictions[:, CAUSE_INDICES], average="micro")
    final_metrics["f1_macro_for_causes"] = f1_score(labels[:, CAUSE_INDICES], predictions[:, CAUSE_INDICES], average="macro")
    
    # Get f1 metrics for emotions
    final_metrics["f1_micro_for_emotions"] = f1_score(labels[:, EMOTION_INDICES], predictions[:, EMOTION_INDICES], average="micro")
    final_metrics["f1_macro_for_emotions"] = f1_score(labels[:, EMOTION_INDICES], predictions[:, EMOTION_INDICES], average="macro")

    # The global f1_metrics
    final_metrics["f1_micro"] = f1_score(labels, predictions, average="micro")
    final_metrics["f1_macro"] = f1_score(labels, predictions, average="macro")
    
    # Classification report
    print("Classification report for global scores: ")
    print(classification_report(labels[:, GLOBAL_SCORE_INDICES], predictions[:, GLOBAL_SCORE_INDICES], zero_division=0))
    print("Classification report for causes: ")
    print(classification_report(labels[:, CAUSE_INDICES], predictions[:, CAUSE_INDICES], zero_division=0))
    print("Classification report for emotions: ")
    print(classification_report(labels[:, EMOTION_INDICES], predictions[:, EMOTION_INDICES], zero_division=0))
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
        
        global_score_loss = torch.nn.functional.cross_entropy(logits[:, GLOBAL_SCORE_INDICES], labels[:, GLOBAL_SCORE_INDICES])
        emotion_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, EMOTION_INDICES], labels[:, EMOTION_INDICES])
        cause_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, CAUSE_INDICES], labels[:, CAUSE_INDICES])
        
        loss = self.group_weights[0] * global_score_loss + self.group_weights[2] * emotion_loss + self.group_weights[1] * cause_loss
        return (loss, outputs) if return_outputs else loss
    
"""
Set training parameters
"""

class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")

training_args = TrainingArguments(
    output_dir="./models/camembert-fine-tuned",
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

# trainer = MultiTaskClassificationTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=ds["train"],
#     eval_dataset=ds["validation"],
#     compute_metrics=compute_metrics,
#     callbacks=[PrinterCallback],
#     group_weights=(0.7, 4, 4)
# )

"""
Train the model
- Should make this manual but not necessary
"""
# trainer.train()
