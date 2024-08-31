import os
import pandas as pd

from proby.shared_logger import shared_logger


def load_data(metadata):
    smiles_list = []
    input_data_folder = metadata["input_data_folder"]
    for file_name in os.listdir(input_data_folder):
        if file_name.endswith('.xlsx'):
            shared_logger.log(f"processing {file_name} ...")
            full_path = os.path.join(input_data_folder, file_name)
            df = pd.read_excel(full_path, dtype=str)
            column_name = [column for column in df.columns if column.lower() == "smiles"][0]
            df.rename(columns={column_name: "SMILES"}, inplace=True)
            column_name = "SMILES"
            df[column_name] = df[column_name].apply(lambda x: x.strip() if isinstance(x, str) else x)
            smiles_list += df[column_name].to_list()
    smiles_list = sorted(list(set(smiles_list)))
    shared_logger.log(f"{len(smiles_list)} smiles in total")
    metadata["smiles_list"] = smiles_list


def get_smiles(df, column_name):
    return set(df[column_name].to_list())


def delete_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("folder doesn't exist")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"deleted {file_path}")
            except OSError as e:
                print(f"error while deleting: {e}")