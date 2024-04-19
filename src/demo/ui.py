import os
import tkinter as tk
from tkinter import Canvas, Frame, Scrollbar, Text, Label, Entry
import imgkit
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from html_render import lime_analyze
from PIL import Image, ImageTk

notes = pd.read_csv("C:\\Users\\xxnan\\Code\\capstone-mayo\\data\\SOCIALHISTORIES.csv")
saved_model_dir = "C:\\Users\\xxnan\\Code\\capstone-mayo\\saved_models\\standard_new"
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
        begin_ui_update(note)
        text_area.insert(tk.END, text)
    print(f"Number: {number}")

labels_text = list(sdoh_to_labels.keys())

#TODO: This function should be called 8 times, once for each SDOH instead of all at once
#TODO: Left-indent label above images
#TODO: Fix the layout

def begin_ui_update(note):
    if note is not None:
        for i in range(8):
            sdoh = labels_text[i]
            update_ui(note, sdoh)

def update_ui(note, sdoh):
    text_label = Label(content_frame, text=f"{sdoh}", font=("Arial", 20))
    image_label = Label(content_frame, text=sdoh)
    model = sdoh_to_models[sdoh]
    current_sdbh_dict, img_path = lime_analyze(note, sdoh, model, tokenizer)
    actual_label = int(true_labels[sdoh])
    actual_label_name = current_sdbh_dict[actual_label]
    pred_label = int(results[sdoh])
    pred_label_name = current_sdbh_dict[pred_label]

    classes_label = Label(content_frame, text=f"Actual: {actual_label_name}, Predicted: {pred_label_name}", font=("Arial", 16))

    if os.path.exists(img_path):
        # imgkit.from_file(img_path, f"{sdoh}.jpg")
        image = Image.open(img_path)
        image = image.crop((0, 0, image.width - 25, image.height - 25))
        photo = ImageTk.PhotoImage(image)
        image_label.image = photo
        image_label.config(image=image_label.image)
    text_label.pack()
    classes_label.pack()
    image_label.pack()

root = tk.Tk()
root.title("SDOH Classifier")
root.resizable(True, True)

# Create a main frame
main_frame = Frame(root)
main_frame.pack(fill="both", expand=1)

# Create a canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side="left", fill="both", expand=1)

# Add a scrollbar to the canvas
my_scrollbar = Scrollbar(main_frame, orient="vertical", command=my_canvas.yview)
my_scrollbar.pack(side="right", fill="y")

# Configure the canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))

# Create another frame inside the canvas
content_frame = Frame(my_canvas)
# Add that new frame to a window in the canvas
my_canvas.create_window((0,0), window=content_frame, anchor="nw")

# Range of 0-7023 for possible notes to try from dataset
label1 = Label(content_frame, text="Enter a number between 0 - 7023")
label1.pack()
entry1 = Entry(content_frame)
entry1.pack()

button = tk.Button(content_frame, text="Submit")
button.pack()
button.config(command=click_button)

# Second part with textarea
label2 = Label(content_frame, text="Text")
label2.pack()

text_area = Text(content_frame, height=5, width=40)
text_area.insert(tk.END, "Display the note corresponding to the note number specified above")
text_area.pack()

load_models()

root.mainloop()

