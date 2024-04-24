from pathlib import Path
import pandas as pd
from helper import set_helper_sdoh, test_train_split
from model import Model

sdoh_to_labels = {
    "sdoh_community_present": 2,
    "sdoh_community_absent": 2,
    "sdoh_education": 2,
    "sdoh_economics": 3,
    "sdoh_environment": 3,
    "behavior_alcohol": 5,
    "behavior_tobacco": 5,
    "behavior_drug": 5
}

base_path = Path(__file__).parent.parent.resolve()
data_path = "./data/SOCIALHISTORIES.csv"

# test_train_split(base_path, data_path)

# for sdoh, num_labels in sdoh_to_labels.items():
#     set_helper_sdoh(sdoh)
#     model = Model(sdoh, num_labels, "emilyalsentzer/Bio_ClinicalBERT", 15, 32, base_path, False, True, output_dir="E:/", cv=False)
#     model.train()

for sdoh, num_labels in sdoh_to_labels.items():
    set_helper_sdoh(sdoh)
    model = Model(sdoh, num_labels, "emilyalsentzer/Bio_ClinicalBERT", 15, 32, base_path, False, False, output_dir="E:/", cv=True)
    model.train()

    