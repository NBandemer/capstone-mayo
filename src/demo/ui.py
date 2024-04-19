import os
import tkinter as tk
from tkinter import Text, Label, Entry
import imgkit
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from html_render import lime_analyze
from PIL import Image, ImageTk

notes = pd.read_csv("C:\\Users\\xxnan\\Code\\capstone-mayo\\data\\SOCIALHISTORIES.csv")
saved_model_dir = "C:\\Users\\xxnan\\Code\\capstone-mayo\\saved_models\\standard"
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
note = None

results = {
    "sdoh_community_present": None,
    "sdoh_community_absent": None,
    "sdoh_education": None,
    "sdoh_economics": None,
    "sdoh_environment": None,
    "behavior_alcohol": None,
    "behavior_tobacco": None,
    "behavior_drug": None
}

true_labels = {
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

def load_models():
    for sdoh, num_labels in sdoh_to_labels.items():
        model =  AutoModelForSequenceClassification.from_pretrained(os.path.join(saved_model_dir, sdoh))
        sdoh_to_models[sdoh] = model

def classify_note(note):
    for sdoh, model in sdoh_to_models.items():
        true_labels[sdoh] = int(note[sdoh])
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, task="text-classification", device=0)
        result = pipeline(note['TEXT'])
        results[sdoh] = result[0]['label'].split("_")[-1]

def click_button():
    number = int(entry1.get())
    if number < 0 or number > 7025:
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, "Number out of range")
    else:
        text_area.delete(1.0, tk.END)
        note = notes.iloc[number]
        text = note['TEXT']
        classify_note(note)
        update_ui(note)
        text_area.insert(tk.END, text)
    print(f"Number: {number}")

labels_text = list(sdoh_to_labels.keys())

def update_ui(note):
    if note is not None:
        for i in range(8):
            sdoh = labels_text[i]
            label = Label(root, text=sdoh)
            model = sdoh_to_models[sdoh]
            img_path = lime_analyze(note, sdoh, model, tokenizer)
            if os.path.exists(img_path):
                # imgkit.from_file(img_path, f"{sdoh}.jpg")
                image = Image.open(img_path)
                photo = ImageTk.PhotoImage(image)
                label.image = photo
                label.config(image=label.image)
            label.pack()

root = tk.Tk()
root.title("UI Example")
root.resizable(True, True)

# Top part with label and text field for a number
label1 = Label(root, text="Enter a number between 0 - 7025")
label1.pack()
entry1 = Entry(root)
entry1.pack()

button = tk.Button(root, text="Submit")
button.pack()
button.config(command=click_button)

# Second part with textarea
label2 = Label(root, text="Text")
label2.pack()

text_area = Text(root, height=5, width=40)
text_area.insert(tk.END, "Display the note corresponding to the note number specified above")
text_area.pack()

# Third section with 8 labels in a 4x2 grid


# Last part rendered from an HTML file
# This part is beyond the scope of Tkinter and would require a different approach or tool
load_models()

root.mainloop()

