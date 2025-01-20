import chemprop
import os
from datetime import datetime
from pathlib import Path

current_date = datetime.now()
timestamp = current_date.strftime("%Y%m%d%H%M%S")

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
save_dir = os.path.join(root_folder_path, f"model_candidates/model_2_{timestamp}")

arguments = [
    '--data_path', os.path.join(processed_data_folder, 'model_2_train_val_data.csv'),
    '--separate_test_path', os.path.join(processed_data_folder, 'model_2_test_full_data.csv'),
    '--dataset_type', 'regression',
    '--save_dir', save_dir,
    '--epochs', '20',
    '--smiles_columns', 'Smiles', 'Solvent',
    '--number_of_molecules', '2',
    '--target_columns', "Scaled Absorption max (nm)", "Scaled Emission max (nm)", "Scaled Lifetime (ns)", "Scaled Quantum yield", "Scaled log(e/mol-1 dm3 cm-1)", "Scaled abs FWHM (cm-1)", "Scaled emi FWHM (cm-1)", "Scaled abs FWHM (nm)", "Scaled emi FWHM (nm)",
    '--save_smiles_splits',
    '--seed', '42',
    '--split_type', 'random_with_repeated_smiles',
    '--bias',  # default false
    '--hidden_size', '1024',   # default 300 1024
    '--depth', '8'  # default 3
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)