#!/bin/bash
#Use this copy to tweak parameters
python script.py \
-m emilyalsentzer/Bio_ClinicalBERT \
-s community_present \
-d "../data/clean/PREPROCESSED-NOTES.csv" \
--eval_batch 16 \
--train_batch 16 \
-l 512 \
-t 0.8 \
-e 4 \
--logs '../logs' \
--save "../saved_models"
