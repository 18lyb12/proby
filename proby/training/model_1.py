import chemprop
import os
from datetime import datetime
from pathlib import Path

current_date = datetime.now()
timestamp = current_date.strftime("%Y%m%d%H%M%S")

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
save_dir = os.path.join(root_folder_path, f"model_candidates/model_1_{timestamp}")

arguments = [
    '--data_path', os.path.join(processed_data_folder, 'model_1_train_val_data.csv'),
    '--features_path', os.path.join(processed_data_folder, 'model_1_train_val_features.csv'),
    '--separate_test_path', os.path.join(processed_data_folder, 'model_1_test_full_data.csv'),
    '--separate_test_features_path', os.path.join(processed_data_folder, 'model_1_test_features.csv'),
    '--dataset_type', 'classification',
    '--class_balance',
    '--save_dir', save_dir,
    '--epochs', '10',  # default 30
    '--smiles_columns', 'smiles',
    '--target_columns', 'new_category',
    '--save_smiles_splits',
    '--split_type', 'random_with_repeated_smiles',
    '--seed', '42',
    '--bias',  # default false
    '--hidden_size', '1024',  # default 300
    '--depth', '5',  # default 3
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
