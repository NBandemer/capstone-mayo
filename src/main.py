from multiple_bert import *
import argparse
from pathlib import Path
import json

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

project_base_path = Path(__file__).parent.parent.resolve()

parser = argparse.ArgumentParser(
    description="Extraction of social determinants of health.")
parser.add_argument("--config", action="store",
                    help="Path to config file")

args = parser.parse_args()
args_dict = vars(args)

invalid = False

if args.config is None:
    print('Missing required argument (config)!')
    exit(1)

json_path = args.config

if not Path(json_path).is_file():
    print('Invalid path to config file!')
    exit(1)

with open(json_path) as json_file:
    args_dict = json.load(json_file)

    if 'sdoh' not in args_dict:
        print('Missing required argument (sdoh)!')
        invalid = True

    if 'num_labels' not in args_dict:
        print('Missing required argument (num_labels)!')
        invalid = True

    if 'model' not in args_dict:
        print('Missing required argument (model)!')
        invalid = True

    if 'epochs' not in args_dict:
        print('Missing required argument (epochs)!')
        invalid = True

    if 'batch' not in args_dict:
        print('Missing required argument (batch)!')
        invalid = True

    if invalid:
        exit(1)

    args = argparse.Namespace(**args_dict)

#print(args)

model_trainer = TrainModel(args.sdoh, int(args.num_labels), args.model, int(args.epochs), int(args.batch), project_base_path)

model_trainer.generate_model()
