from multiple_bert import *
import argparse

# sdoh_list = {
#     "sdoh_community_present": 2,
#     "sdoh_community_absent": 2,
#     "sdoh_education": 2,
#     "sdoh_economics": 2,
#     "sdoh_environment": 2,
#     "behavior_alcohol": 5,
#     "behavior_tobacco": 5,
#     "behavior_drug": 5
# }

parser = argparse.ArgumentParser(
    description="Extraction of social determinants of health.")
parser.add_argument("--model", action="store",
                    help="Pre-trained model name from HuggingFace pipeline")
parser.add_argument("--sdoh", action="store",
                    help=f"Social determinant of health to extract.")
parser.add_argument("--num_labels", action="store",
                    help="Number of labels for training")
parser.add_argument("--data", action="store",
                    help="Path to dataset")
parser.add_argument("--batch", action="store",
                    help="Batch size for model")
parser.add_argument("--epochs", action="store",
                    help="Number of epochs for training")

args = parser.parse_args()
args_dict = vars(args)

invalid = False

for key, value in args_dict.items():
    if value is None:
        invalid = True
        print(f'Missing required argument ({key})!')

if invalid:
    exit(1)

# print(f"Model: {args.model}", f"SDOH: {args.sdoh}", f"Labels: {args.num_labels}", f"Epochs: {args.epochs}", f"Batch: {args.batch}",  sep="\n")
# for sdoh, num_label in sdoh_list.items():
model_trainer = TrainModel(args.sdoh, int(args.num_labels), args.model, int(args.epochs), int(args.batch))

model_trainer.generate_model()
