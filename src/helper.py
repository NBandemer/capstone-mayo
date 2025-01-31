from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

import os
import pandas as pd

current_sbdh = "sdoh_education"

sbdh_substance = {
    0: 'None',
    1: 'Present',
    2: 'Past',
    3: 'Never',
    4: 'Unsure'
}

#Economics (employed) classifications
sbdh_econ_env = {
    0: 'None',
    1: 'True',
    2: 'False',
}

#Community or Education classifications
sbdh_community_ed = {
    0: 'False',
    1: 'True',
}

def set_sdoh(sdoh_name):
    """
    This function sets the SDOH described in the classification report
    """
    global current_sbdh
    current_sbdh = sdoh_name

def balance_data(df):
    values = df['y'].value_counts()
    majority = df[df['y'] == values.idxmax()]
    desired_samples = len(majority)

    for label in values.index:
        if label == values.idxmax():
            continue
        minority = df[df['y'] == label]
        upsampled_minority = resample(minority,
                                      replace=True,  # Sample with replacement
                                      n_samples=desired_samples,  # Match number of majority class
                                      random_state=42)
        majority = pd.concat([majority, upsampled_minority])

    return majority


def test_train_split(base_path, data):
    """
    This function creates the test_train_split for all the SDoH from the pre processed data
    """
    dataset = pd.read_csv(data)

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

    # Iterate through each SDOH data category
    for category, data in sdoh_data.items():
        # Create folder for each category
        category_data_path = f"{base_path}/data/test_train_split/{category}"
        os.makedirs(category_data_path, exist_ok=True)

        # Split data for the current category
        X_train, X_val, y_train, y_val = train_test_split(
            text_data, data, random_state=0, train_size=0.8, stratify=data
        )

        # Save train and test data to seprate csvs
        pd.DataFrame({"text": X_train, category: y_train}).to_csv(f"{category_data_path}/train.csv", index=False)
        pd.DataFrame({"text": X_val, category: y_val}).to_csv(f"{category_data_path}/test.csv", index=False)


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
    report = classification_report(labels, preds, output_dict=True)
    auc = roc_auc_score(labels, preds, average='weighted', multi_class='ovr')

    # Confusion Matrix
    cm = ConfusionMatrixDisplay.from_predictions(labels, preds)
    cm.plot()
    plt.show()
    
    if current_sbdh.startswith("behavior"):
        current_sbdh_dict = sbdh_substance
    elif current_sbdh == "sdoh_economics" or current_sbdh == "sdoh_environment":
        current_sbdh_dict = sbdh_econ_env
    else:
        current_sbdh_dict = sbdh_community_ed
    
    for key, value in current_sbdh_dict.items():
        report[value] = report[str(key)]
        del report[str(key)]
    
    print(f'Classification Report for {current_sbdh}', report, sep='\n')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
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
