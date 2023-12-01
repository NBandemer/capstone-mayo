import argparse
import data as d
import model as m

def main():
    parser = argparse.ArgumentParser(description="Extraction of social determinants of health.")
    parser.add_argument("-m", "--model_name", action="store", help="Pre-trained model name from HuggingFace pipeline")
    parser.add_argument("-s", "--social_determinant", action="store", help=f"Social determinant of health to extract.", choices=d.all_sd)
    parser.add_argument("-d", "--data_path", action="store", help="Path to dataset")
    parser.add_argument("--train_batch", action="store", help="Batch size for model training")
    parser.add_argument("--eval_batch", action="store", help="Batch size for model evaluation")
    parser.add_argument("-l", "--max_seq_len", action="store", help="Max sequence length for model")
    parser.add_argument("-t", "--train_size", action="store", help="Percentage of data to be used for training")
    parser.add_argument("-e", "--num_epochs", action="store", help="Number of epochs for training")
    parser.add_argument("--logs", action="store", help="Path for tensor and epoch logs")
    parser.add_argument("--save", action="store", help="Directory for saved models")

    args = parser.parse_args()
    args_dict = vars(args)

    invalid = False

    for key, value in args_dict.items():
        if value is None:
            invalid = True
            print(f'Missing required argument ({key})!')
    
    if invalid: exit(1)

    model_name = args.model_name
    social_determinant = args.social_determinant
    dataset_path = args.data_path
    max_seq_len = int(args.max_seq_len)
    train_size = float(args.train_size)
    num_epochs = int(args.num_epochs)
    train_batch_size = int(args.train_batch)
    eval_batch_size = int(args.eval_batch)
    log_path = args.logs
    save_directory = args.save

    data = load_data(
        path=dataset_path,
        social=social_determinant,
        max_seq_len=max_seq_len
    )
    model = load_model(model_name)
    data.encode_data(model, train_size)
    model.set_training_args(
        epochs=num_epochs, 
        log_path=log_path, 
        decay=1e-5,
        eval_batch_size=eval_batch_size,
        train_batch_size=train_batch_size
    )
    model.set_trainer(
        train=data.train_dataset,
        val=data.val_dataset
    )
    model.train()
    model.evaluate()
    model.save_model(save_directory)

def load_data(path, social, max_seq_len):
    data = d.Data(
        social=social,
        path=path,
        max_seq_len=max_seq_len
    )
    return data

def load_model(modelName):
    model = m.Model(modelName)
    return model
    
if __name__ == "__main__":
    main()
    