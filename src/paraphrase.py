from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import pandas as pd
import medspacy
import re
import torch

device = "cuda"
# tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
# model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').to(device)

# ! humarin/chatgpt_paraphraser_on_T5_base: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
def paraphrase(
    question,
    num_beams=10,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f"{question}",
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

def paraphrase_and_combine(row):
    # Split each part into sentences based on ';' and '.'
    sentences = re.split('; |\\. |:', row)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Paraphrase each sentence and combine them
    paraphrased_sentences = [' '.join(get_response(sentence)) for sentence in sentences]

    # Combine the paraphrased parts into one row
    return ' '.join(paraphrased_sentences)

def get_response(input_text,num_return_sequences=1,num_beams=5):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

# df = pd.read_csv("C:\\Users\\manav\\OneDrive\\Desktop\\capstone-mayo\\data\\SOCIALHISTORIES.csv")

# index = 2
# print(df['TEXT'][index])
# print("---------------------------------------- New Text --------------------------------------------")
# print(paraphrase_and_combine(df['TEXT'][index]))