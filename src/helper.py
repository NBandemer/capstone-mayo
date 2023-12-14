from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

# Function to save metrics to a CSV file
def save_metrics_to_csv(json_filepath, csv_filename):
    with open(json_filepath) as file:
        data = json.load(file)

    # Extract the 'log_history' column
    log_history = data['log_history']

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(log_history)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

# Function to plot metrics
def plot_metrics_from_csv(csv_filepath, output_dir):
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