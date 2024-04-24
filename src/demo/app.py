import streamlit as st
import pandas as pd
from util import preprocess_text, load_models, lime_analyze, classify_note, pred_labels, base_path

# Load the models
load_models()

# Initialize the results
results = pd.DataFrame(columns=['Predicted Label', 'Explain'])

# Create a header with a title and explanatory text
st.title('Classifying SDOHs in Discharge Summaries')
st.markdown('This is a brief explanation of your project. You can provide more details here.')

# Create a text area for the note input
note = st.text_area('Enter note here', '')

# Create a button for the classification
if st.button('Classify'):
    # Perform the classification here and update the 'results' DataFrame
    # For example:
    text = preprocess_text(note)
    classify_note(text)

    for sdoh, label in pred_labels.items():
        results.loc[sdoh] = [label, 'Explain']

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

# Display the results DataFrame


# Create buttons for explanations and display the explanation when a button is clicked
# for index, row in results.iterrows():
#     row['Explain'] = st.button(f'Explain', key=index)

#     if row['Explain']:
#         lime_analyze(note, index)
#         path = get_path_to_explanation(index)
#         st.image(path)

# Create a footer with acknowledgments
st.markdown('---')
st.markdown('**Acknowledgments:** This is where you can acknowledge the contributions of others to your project.')
