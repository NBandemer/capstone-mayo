import datetime
import os
import pandas as pd
import json
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

def save_metrics_to_csv(json_filepath, csv_filename):
    with open(json_filepath) as file:
        data = json.load(file)

        log_history = data['log_history'] #focus on this column for history
        df = pd.DataFrame(log_history) # Convert the list of dictionaries to a DataFrame

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

    
