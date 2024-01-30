from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

import os
import pandas as pd

def test_train_split():
    """
    This function creates the test_train_split for all the SDoH from the pre processed data
    """
    dataset = pd.read_csv("data\PREPROCESSED-NOTES.csv")

    text_data = dataset["text"].to_list()

    sdoh_data = {
        "sdoh_community_present": dataset["sdoh_community_present"].to_list(),
        "sdoh_community_absent": dataset["sdoh_community_absent"].to_list(),
        "sdoh_education": dataset["sdoh_education"].to_list(),
        "sdoh_economics": dataset["sdoh_economics"].to_list(),
        "sdoh_environment": dataset["sdoh_environment"].to_list(),
        "behavior_alcohol": dataset["behavior_alcohol"].to_list(),
        "behavior_tobacco": dataset["behavior_tobacco"].to_list(),
        "behavior_drug": dataset["behavior_drug"].to_list()
    }

    base_path = 'test_train_split/behavior_drug'
    os.makedirs(base_path, exist_ok=True)

    # Iterate through each SDOH data category
    for category, data in sdoh_data.items():
        # Create folder for each category
        base_path = f"test_train_split/{category}"
        os.makedirs(base_path, exist_ok=True)

        # Split data for the current category
        X_train, X_val, y_train, y_val = train_test_split(
            text_data, data, random_state=0, train_size=0.8, stratify=data
        )

            # Save all splits as CSV files
        pd.DataFrame({"text": X_train}).to_csv(f"{base_path}/X_train.csv", index=False)
        pd.DataFrame({"text": X_val}).to_csv(f"{base_path}/X_val.csv", index=False)
        pd.DataFrame({category: y_train}).to_csv(f"{base_path}/y_train.csv", index=False)
        pd.DataFrame({category: y_val}).to_csv(f"{base_path}/y_val.csv", index=False)

def compute_metrics(eval_pred):
    """
    Calculates the metrics at the end of each epoch
    """
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

def plot_metric_from_tensor(log_dir, save_dir):
    '''
    
    '''
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    graph1_data = event_acc.Scalars("eval/loss")
    graph2_data = event_acc.Scalars("train/loss")

    # Access step and value directly from events
    steps1 = [event.step for event in graph1_data]
    values1 = [event.value for event in graph1_data]

    steps2 = [event.step for event in graph2_data]
    values2 = [event.value for event in graph2_data]

    plt.figure(figsize=(10, 6))
    plt.plot(steps1, values1, label="Eval Loss")
    plt.plot(steps2, values2, label="Train Loss")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Combined Graphs")
    # plt.show()

    # Save the graph to the specified folder
    plt.savefig(f"{save_dir}")
