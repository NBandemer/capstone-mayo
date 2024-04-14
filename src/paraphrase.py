from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pandas as pd
import medspacy
import re

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

# ! humarin/chatgpt_paraphraser_on_T5_base: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=900
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
    # Split the row into two parts based on 'social history:'
    parts = row.split('social history: ')
    parts = [part.strip() for part in parts if part.strip()]

    paraphrased_parts = []
    for part in parts:
        # Split each part into sentences based on ';' and '.'
        sentences = re.split('; |\\. ', part)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Paraphrase each sentence and combine them
        paraphrased_sentences = [' '.join(paraphrase(sentence)) for sentence in sentences]
        paraphrased_part = '; '.join(paraphrased_sentences)

        # Add 'social history:' back to the paraphrased part
        paraphrased_parts.append('social history: ' + paraphrased_part)

    # Combine the paraphrased parts into one row
    return '  '.join(paraphrased_parts)

def para_whole(row):
    row = str(row)
    generic_prompt = f"Paraphrase the following medical discharge summary while preserving any information relevant to the patient's social determinants of health, such as education, economics, environment, and substance use (alcohol, tobacco, and drug use). The note should keep a similar format and style as the original, but can swap words and phrases around and use synonyms: {row}"

    sentence = row

    return paraphrase(sentence)

df = pd.read_csv("C:\\Users\\manav\\OneDrive\\Desktop\\capstone-mayo\\data\\SOCIALHISTORIES.csv")

print(df['TEXT'][2])
print("---------------------------------------- New Text --------------------------------------------")
# print(paraphrase_and_combine(df['TEXT'][2]))
print(para_whole(df['TEXT'][2]))