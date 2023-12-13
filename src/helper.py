from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to save metrics to a CSV file
def save_metrics_to_csv(metrics, csv_filename):
    df = pd.DataFrame(metrics)
    df.to_csv(csv_filename, index=False)

# Function to plot metrics
def plot_metrics(metrics, output_dir):
    plt.figure(figsize=(10, 6))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(metrics['step'], metrics['eval_accuracy'], label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Step')
    plt.legend()

    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(metrics['step'], metrics['eval_precision'], label='Precision')
    plt.title('Precision')
    plt.xlabel('Step')
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(metrics['step'], metrics['eval_recall'], label='Recall')
    plt.title('Recall')
    plt.xlabel('Step')
    plt.legend()

    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(metrics['step'], metrics['eval_f1'], label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Step')
    plt.legend()

    plt.tight_layout()
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