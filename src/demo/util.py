import os
from pathlib import Path
import re
import imgkit
from lime.lime_text import LimeTextExplainer
import torch
import medspacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from PIL import Image

base_path = Path(__file__).parent.parent.parent.resolve()

saved_model_dir = f"{base_path}/saved_models/standard_new"

nlp = medspacy.load()
sectionizer = nlp.add_pipe("medspacy_sectionizer")

MAX_LENGTH = 128
model = None
tokenizer = None

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

pred_labels = {
    "sdoh_community_present": 0,
    "sdoh_community_absent": 0,
    "sdoh_education": 0,
    "sdoh_economics": 0,
    "sdoh_environment": 0,
    "behavior_alcohol": 0,
    "behavior_tobacco": 0,
    "behavior_drug": 0
}

sdoh_to_models = {
    "sdoh_community_present": None,
    "sdoh_community_absent": None,
    "sdoh_education": None,
    "sdoh_economics": None,
    "sdoh_environment": None,
    "behavior_alcohol": None,
    "behavior_tobacco": None,
    "behavior_drug": None
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

sdohs = list(sdoh_to_labels.keys())
sdoh_class_names = []
current_sbdh_dict = None

def classify_note(note):
    for sdoh, model in sdoh_to_models.items():
        (model, tokenizer) = model
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, task="text-classification", device=0)
        result = pipeline(note)
        pred_labels[sdoh] = result[0]['label'].split("_")[-1]

def init(sdoh, m, t):
    global sdoh_class_names, model, tokenizer, current_sbdh_dict

    model = m
    tokenizer = t

    if sdoh.startswith("behavior"):
        current_sbdh_dict = sbdh_substance
    elif sdoh == "sdoh_economics" or sdoh == "sdoh_environment":
        current_sbdh_dict = sbdh_econ_env
    else:
        current_sbdh_dict = sbdh_community_ed

    sdoh_class_names = list(current_sbdh_dict.values())

def load_models():
    global tokenizer
    for sdoh, _ in sdoh_to_labels.items():
        model =  AutoModelForSequenceClassification.from_pretrained(os.path.join(saved_model_dir, sdoh))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(saved_model_dir, sdoh))
        sdoh_to_models[sdoh] = (model, tokenizer)

def predict_proba(text):
    global model, tokenizer

    if model is None or tokenizer is None:
        print("No model or tokenizer")
        exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        model.to(device)
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt').to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()
    
    return probs

def preprocess_text(text):
    doc = nlp(text)
    if 'social_history' not in doc._.section_categories:
        return text
    
    # We only consider social history and past medical history sections
    sections = doc._.sections
    relevant_sections = [section for section in sections if section.category == "social_history" or section.category == "past_medical_history"]

    # TODO: Manually check more notes, consider different patterns
    
    for index, section in enumerate(sections):
        if section.category == "social_history":
            social_index = index
            start_index = section.body_start
            end_index = section.body_end
            social_history += " " if social_history != "" else ""
            social_history += str(doc[start_index:end_index])
        
        # Only consider past medical history sections that are immediately after social history
        # Add the keyword that triggered this section back into the text, as it often is the word "history" or similar
        elif section.category == "past_medical_history" and index - 1 == social_index:
            start_index = section.body_start
            end_index = section.body_end
            key_word = section.rule.literal + " "
            social_history += " " + key_word + str(doc[start_index:end_index])

    # Remove newlines, carriage returns, and tabs
    processed_social_history = re.sub(r'[\n\r\t]+', ' ', social_history).strip().lower()
    return processed_social_history

def lime_analyze(text, sdoh):
    (model, tokenizer) = sdoh_to_models[sdoh]

    init(sdoh, model, tokenizer)

    pred_label = predict_proba(text).argmax()

    # true_label_name = current_sbdh_dict[true_label]
    # pred_label_name = current_sbdh_dict[pred_label]

    labels_to_explain = (pred_label,)

    num_features = 5 if len(text.split()) >= 5 else len(text.split())

    explainer = LimeTextExplainer(class_names=sdoh_class_names)
    explanation = explainer.explain_instance(text, predict_proba, num_features=num_features, num_samples=500, labels=labels_to_explain)
    
    explanation.save_to_file(f'{base_path}/src/demo/{sdoh}.html')
    path = render_lime(sdoh)
    return current_sbdh_dict, path

def render_lime(sdoh):
    # Specify the path to your HTML file
    path_to_html_file = f"{base_path}/src/demo/{sdoh}.html"

    # Specify the path where you want to save the image
    path_to_output_image = f"{base_path}/src/demo/{sdoh}.jpg"

    # Convert the HTML file to an image
    imgkit.from_file(path_to_html_file, path_to_output_image)

    image = Image.open(path_to_output_image)
    image = image.crop((0, 0, image.width - 25, image.height - 25))
    image.save(path_to_output_image)
    
    return path_to_output_image