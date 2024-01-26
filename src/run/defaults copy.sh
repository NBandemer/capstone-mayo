#!/bin/bash
#Use this copy to tweak parameters
python ../scripts/script.py \
--model emilyalsentzer/Bio_ClinicalBERT \
--sdoh community_present \
--data "../../data/clean/PREPROCESSED-NOTES-NEW.csv" \
--batch 8 \
--train_size 0.8 \
--epochs 4 \
--logs '../logs' \
--save "../saved_models"
