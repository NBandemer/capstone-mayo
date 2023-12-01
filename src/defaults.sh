#!/bin/bash
echo "Running script with defaults!" 
python script.py \
-m emilyalsentzer/Bio_ClinicalBERT \
-s community_present \
-d "../data/clean/PREPROCESSED-NOTES.csv" \
-l 512 \
-t 0.8 \
-e 4 \
--logs '../logs' \
--save "../saved_models"

