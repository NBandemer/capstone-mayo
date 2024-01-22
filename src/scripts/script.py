import argparse
import sys

sys.path.insert(1, '../util')

import data as d
import model as m


def main():
    # CLI

    parser = argparse.ArgumentParser(
        description="Extraction of social determinants of health.")
    parser.add_argument("--model", action="store",
                        help="Pre-trained model name from HuggingFace pipeline")
    parser.add_argument("--sdoh", action="store",
                        help=f"Social determinant of health to extract.", choices=d.all_sd)
    parser.add_argument("--data", action="store",
                        help="Path to dataset")
    parser.add_argument("--batch", action="store",
                        help="Batch size for model")
    parser.add_argument("--train_size", action="store",
                        help="Percentage as decimal of data to be used for training")
    parser.add_argument("--epochs", action="store",
                        help="Number of epochs for training")
    parser.add_argument("--logs", action="store",
                        help="Path for tensor and epoch logs")
    parser.add_argument("--save", action="store",
                        help="Directory for saved models")

    args = parser.parse_args()
    args_dict = vars(args)

    invalid = False

    for key, value in args_dict.items():
        if value is None:
            invalid = True
            print(f'Missing required argument ({key})!')

    if invalid:
        exit(1)
    
    # Initialize data and model
    data = d.Data(
        path=args.data,
        sdoh=args.sdoh,
        train_size=float(args.train_size)
    )

    model = m.Model(
        model=args.model,
        epochs=int(args.epochs),
        batch=int(args.batch),
        logs=args.logs,
        save_dir=args.save,
        data=data
    )

    model.run()


if __name__ == "__main__":
    main()
