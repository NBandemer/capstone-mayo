import os
from lime.lime_text import LimeTextExplainer
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import torch

device = 'cuda'

def predict_proba(texts, tokenizer, model):
    model.to(device)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = len(inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    return probabilities, tokens

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

project_base_path = Path(__file__).parent.parent.resolve()

for sdoh in sdoh_to_labels.keys():
    if sdoh.startswith("behavior"):
        current_sbdh_dict = sbdh_substance
    elif sdoh == "sdoh_economics" or sdoh == "sdoh_environment":
        current_sbdh_dict = sbdh_econ_env
    else:
        current_sbdh_dict = sbdh_community_ed

    sdoh_class_names = list(current_sbdh_dict.values())
    explainer = LimeTextExplainer(class_names=sdoh_class_names)

    # Load the data
    data_path = os.path.join(project_base_path, 'data', 'test_train_split', sdoh, 'test.csv')
    data = pd.read_csv(data_path)
    
    # Load the model
    saved_models_dir = os.path.join(project_base_path, f'saved_models/standard_new/{sdoh}')
    model = AutoModelForSequenceClassification.from_pretrained(saved_models_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_models_dir)
    
    sample = data.sample(1)
    text = sample['text'].tolist()

    probs, tokens = predict_proba(text, tokenizer, model)

    explanation = explainer.explain_instance(text, probs, num_features=tokens)
    print(explanation.as_list())

    break

