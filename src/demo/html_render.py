import imgkit
from lime.lime_text import LimeTextExplainer
import torch

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

sdoh_class_names = []
current_sbdh_dict = None

def init(sdoh, m, t):
    global sdoh_class_names, model, tokenizer

    model = m
    tokenizer = t

    if sdoh.startswith("behavior"):
        current_sbdh_dict = sbdh_substance
    elif sdoh == "sdoh_economics" or sdoh == "sdoh_environment":
        current_sbdh_dict = sbdh_econ_env
    else:
        current_sbdh_dict = sbdh_community_ed

    sdoh_class_names = list(current_sbdh_dict.values())

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

def lime_analyze(sample, sdoh, model, tokenizer):
    init(sdoh, model, tokenizer)
    text = str(sample['TEXT'])
    true_label = sample[sdoh]
    pred_label = predict_proba(text).argmax()

    # true_label_name = current_sbdh_dict[true_label]
    # pred_label_name = current_sbdh_dict[pred_label]

    if pred_label != true_label:
        labels_to_explain = (pred_label,true_label)
    else:
        labels_to_explain = (pred_label,)

    num_features = 5 if len(text.split()) >= 5 else len(text.split())

    explainer = LimeTextExplainer(class_names=sdoh_class_names)
    explanation = explainer.explain_instance(text, predict_proba, num_features=num_features, num_samples=500, labels=labels_to_explain)
    
    explanation.save_to_file(f'C:\\Users\\xxnan\\Code\\capstone-mayo\\src\\demo\\{sdoh}.html')
    return render_lime(sdoh)

def render_lime(sdoh):
    # Specify the path to your HTML file
    path_to_html_file = f"C:\\Users\\xxnan\\Code\\capstone-mayo\\src\\demo\\{sdoh}.html"

    # Specify the path where you want to save the image
    path_to_output_image = f"C:\\Users\\xxnan\\Code\\capstone-mayo\\src\\demo\\{sdoh}.jpg"

    # Convert the HTML file to an image
    imgkit.from_file(path_to_html_file, path_to_output_image)
    return path_to_output_image