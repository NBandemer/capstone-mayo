from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

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

def save_metrics_to_csv(json_filepath, csv_filename):
    """
    Function to save metrics to a CSV file
    """
    with open(json_filepath) as file:
        data = json.load(file)

    # Extract the 'log_history' column
    log_history = data['log_history']

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(log_history)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

def plot_metrics_from_csv(csv_filepath, output_dir):
    """
    Function to plot metrics
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filepath)

    plt.figure(figsize=(10, 6))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['eval_accuracy'], label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()

    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['eval_precision'], label='Precision')
    plt.title('Precision')
    plt.xlabel('epoch')
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['eval_recall'], label='Recall')
    plt.title('Recall')
    plt.xlabel('epoch')
    plt.legend()

    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(df['epoch'], df['eval_f1'], label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('epoch')
    plt.legend()

    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.show()

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

def get_latest_checkpoint(folder_path):
    """
    Returns the latest checkpoint from the ephoc logs to convert the metrics into a csv to make it more readable
    """
    # Get a list of all files and directories in the specified folder
    files_and_dirs = os.listdir(folder_path)

    # Filter only directories (assumed to be checkpoints)
    checkpoint_dirs = [d for d in files_and_dirs if os.path.isdir(os.path.join(folder_path, d))]

    if not checkpoint_dirs:
        print("No checkpoint directories found.")
        return None

    # Extract the checkpoint numbers from the directory names
    checkpoint_numbers = [int(d.split('-')[1]) for d in checkpoint_dirs]

    # Identify the directory with the highest checkpoint number
    latest_checkpoint = os.path.join(folder_path, f"checkpoint-{max(checkpoint_numbers)}")

    return latest_checkpoint