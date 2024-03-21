from model import *
import argparse
from pathlib import Path
import json

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

project_base_path = Path(__file__).parent.parent.resolve()

parser = argparse.ArgumentParser(
    description="Extraction of social determinants of health.")
parser.add_argument("--config", action="store",
                    help="Path to config file")
parser.add_argument("-e", "--eval", action="store_true", help="Flag to indicate test/evaluation mode")
parser.add_argument("-s", "--split", action="store_true", help="Flag to indicate data split mode, necessary for new data")

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

# Data Split Mode for new data
if args.split:
    if os.path.isfile(config.data):
        test_train_split(project_base_path, config.data)
    else:
        print('Invalid data file path: ', config.data)
    exit(1)

# for sdoh in sdoh_list:
    # num_labels = sdoh_list[sdoh]
if config.output:
    config.output = Path(config.output)

# Train all models
# for sdoh, labels in sdoh_to_labels:
    # model.train()
    
model = Model(config.sdoh, sdoh_to_labels[config.sdoh], config.model, int(config.epochs), int(config.batch), project_base_path, bool(config.balanced), bool(config.weighted), output_dir=config.output, cv=bool(config.cv))

if args.eval: #evaluation mode
    model.test()
else:
    model.train()

# Cross Val all models
# for (sdoh, labels) in sdoh_to_labels.items():
#     model = Model(sdoh, labels, config.model, int(config.epochs), int(config.batch), project_base_path, bool(config.balanced), bool(config.weighted), output_dir=config.output, cv=True)
#     model.train()
