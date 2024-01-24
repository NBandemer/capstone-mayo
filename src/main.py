from multiple_bert import *

sdoh_list = {
    "sdoh_community_present": 2,
    "sdoh_community_absent": 2,
    "sdoh_education": 2,
    "sdoh_economics": 2,
    "sdoh_environment": 2,
    "behavior_alcohol": 5,
    "behavior_tobacco": 5,
    "behavior_drug": 5
}

print()

for sdoh, num_label in sdoh_list.items():
    model_trainer = TrainModel(sdoh, num_label)

    model_trainer.generate_model()
