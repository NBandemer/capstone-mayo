import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from paraphrase import paraphrase_and_combine

def synthesize_data(original_data, sdoh_name):
    """
    This function paraphrases the data to increase the size of the dataset
    """
    df_synth = original_data.copy()
    count = 1
    total = len(original_data)

    for _, row in original_data.iterrows():
        text = str(row['text'])
        class_val = row[sdoh_name]

        # generic_prompt = "Paraphrase the following medical discharge summary while preserving any information relevant to the patient's social determinants of health, such as education, economics, environment, and substance use (alcohol, tobacco, and drug use). The note should keep a similar format and style as the original, but can swap words and phrases around and use synonyms. \n\n"
        
        synth_texts = []

        # for _ in range(num_synths):
        
        synth_text = paraphrase_and_combine(text)

        new_row = {'text': synth_text, sdoh_name: class_val}
        synth_texts.append(new_row)

        new_df = pd.DataFrame(synth_texts)
        df_synth = pd.concat([df_synth, new_df], ignore_index=True)
        print(f"{count}/{total}")
        count += 1

    return df_synth


imbalanced_sdoh_classes = {
    'behavior_alcohol': [2,4],
    'behavior_drug': [1,2,4],
    'behavior_tobacco': [4],
    'sdoh_community_absent': [1],
    'sdoh_economics': [1],
    'sdoh_education': [1],
    'sdoh_environment': [2],
}

for sdoh_name, imbalanced_class_indices in imbalanced_sdoh_classes.items():
    # Load the data
    data_path = f"./data/test_train_split/{sdoh_name}"
    train_data = pd.read_csv(f"{data_path}/train.csv")

    # Select the data that belongs to the imbalanced classes
    imbalanced_data = train_data[train_data[sdoh_name].isin(imbalanced_class_indices)]

    # Create synthetic data (default 2 synthetic rows) for each row belong to the imbalanced classes
    synthesized_data = synthesize_data(imbalanced_data, sdoh_name)
    # test_data = pd.read_csv(f"{data_path}/test.csv")

    # Save the resulting synthetic + original data
    synthesized_data.to_csv(f"{data_path}/train_synthesized.csv", index=False)
    # test_data_balanced.to_csv(f"{data_path}/test_synthesized.csv", index=False)

