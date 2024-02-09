from model import *
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
parser.add_argument("-e", "--eval", action="store_true", help="Flag to indicate test/evaluation mode")

args = parser.parse_args()
args_dict = vars(args)

invalid = False

if args.config is None:
    print('Missing required argument (config)!')
    exit(1)

json_path = args.config

if not os.path.isfile(json_path):
    print('Invalid path to config file!')
    exit(1)

with open(json_path) as json_file:
    config = json.load(json_file)

    if 'sdoh' not in config:
        print('Missing required argument (sdoh)!')
        invalid = True

    if 'num_labels' not in config:
        print('Missing required argument (num_labels)!')
        invalid = True

    if 'model' not in config:
        print('Missing required argument (model)!')
        invalid = True

    if 'epochs' not in config:
        print('Missing required argument (epochs)!')
        invalid = True

    if 'batch' not in config:
        print('Missing required argument (batch)!')
        invalid = True

    if invalid:
        exit(1)

    config = argparse.Namespace(**config)

model = Model(config.sdoh, int(config.num_labels), config.model, int(config.epochs), int(config.batch), project_base_path)

if args.eval: #evaluation mode
    model.test()
else:
    model.train()
