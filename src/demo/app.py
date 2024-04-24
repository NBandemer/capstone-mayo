import streamlit as st
import pandas as pd
from util import preprocess_text, load_models, lime_analyze, classify_note, pred_labels, base_path
from util import *

def convert_numbers_to_labels(sdoh, label):
    label = int(label)
    if sdoh in ['sdoh_community_present', 'sdoh_community_absent', 'sdoh_education']:
        return sbdh_community_ed[label]
    elif sdoh in ['sdoh_economics', 'sdoh_environment']:
        return sbdh_econ_env[label]
    elif sdoh in ['behavior_alcohol', 'behavior_tobacco', 'behavior_drug']:
        return sbdh_substance[label]

# Load the models
load_models()

# Initialize the results
results = pd.DataFrame(columns=['Predicted Label', 'Explain'])

# Create a header with a title and explanatory text
st.title('Classifying SDOHs from Discharge Summaries')
st.markdown('This is a demo for our Capstone project. You can enter a medical discharge summary below and it would show us the predicted labels for each SDoH. It also shows us the LIME analysis which visually demmonstrates how the model came to the predict the label')

# Create a text area for the note input
note = st.text_area('Enter note here', '')

# Create a button for the classification
if st.button('Classify'):
    # Perform the classification here and update the 'results' DataFrame
    text = preprocess_text(note)
    classify_note(text)

    for sdoh, label in pred_labels.items():
        num_to_label = convert_numbers_to_labels(sdoh, label)
        results.loc[sdoh] = [num_to_label, 'Explain']

    st.write(results)

    for sdoh, label in pred_labels.items():
        lime_analyze(note, sdoh)
        path = f"{base_path}/src/demo/{sdoh}.jpg"

        sdoh_label = f'<h2 style="font-family:Courier; color:Blue;">{sdoh}</h2>'
        sdoh_desc = f'<p style="font-family:Courier;">{sdoh} description</p>'

        st.markdown(sdoh_label, unsafe_allow_html=True)
        st.markdown(sdoh_desc, unsafe_allow_html=True) 
        st.image(path)

        st.markdown('<hr style="border:4px solid gray">', unsafe_allow_html=True)

# Create a footer with acknowledgments
st.markdown('---')
st.markdown('**Acknowledgments:** We would like to thank Dr. Imon Banerjee and Dr. Amara Tariq for all their support through out this project.')
