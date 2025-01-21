import os
import pandas as pd

import threading
import time

from collections import deque
from datetime import datetime


class PredictionWithProgress:
    def __init__(self):
        self.stop_flag = threading.Event()

    def print_wait_message(self, prefix):
        while not self.stop_flag.is_set():
            shared_logger.log(f"{prefix}, please wait...")  # Overwrite the line
            time.sleep(5)  # Print every 5 seconds


class SharedLogger:
    def __init__(self, max_size=200):
        self.log_messages = deque(maxlen=max_size)

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} - {message}'
        self.log_messages.append(log_entry)

    def get_logs(self):
        return list(self.log_messages)


# Create a singleton instance of SharedLogger
shared_logger = SharedLogger()


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


prompt_text = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣄⡀⠀⠀⢠⣄⠀⠀⠀⠀⠀⠲⡆⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⢉⣀⣟⠀⣀⠀⠘⡇⠀⠀⠀⠠⠴⠖⢺⠃⢠⡔⠋⠉⠀⠀⠀⠀⠈⠛⠉⢹⠏⢉⣉⡉⠀⠀⠀⠀⠀⠀⠀⠐⠒⣿⠍⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⢹⡇⣀⠀⢸⠃⠀⡇⠀⠀⣀⣀⠼⣤⠷⠒⢸⡧⢤⡶⠖⠂⠀⠀⠀⠀⢻⠉⢩⡁⠘⡏⠀⠀⠀⠀⠀⢄⣀⠀⣴⠥⢤⡴⠖⢲⣦⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⡾⡗⠸⡏⢹⡇⢸⠀⠠⡇⠀⠀⠀⠤⠖⢺⣂⠁⢸⠀⢸⡇⠀⠀⠀⠀⠀⠀⣻⠀⢸⠁⠀⡇⠀⠀⠀⠀⠀⠘⡇⠀⢹⠒⠊⡇⠀⢸⡏⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⠜⠀⡇⢸⡣⢼⠇⠀⠀⢠⡇⠀⠀⠀⡼⠀⢸⠘⢃⡟⠀⢸⡇⠀⠀⠀⠀⠀⠀⠿⢀⡟⢀⡀⡇⠀⠀⠀⠀⠀⠀⡟⠀⢸⠛⢣⠇⠀⣼⠇⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠇⠀⠀⠀⠢⣼⡇⠀⠀⠀⠁⠈⠛⠀⠊⠀⠀⠘⠇⠀⠀⠀⠀⠀⢀⡠⠞⠀⠀⠙⢷⡄⠀⠀⠀⠀⠀⠟⠀⠈⠉⠉⠉⠹⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⢠⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀
⠠⠤⠤⠤⠶⠴⠶⣶⠒⠛⠛⠂⠀⠠⣤⣤⠤⠤⢴⡶⠶⠒⠛⠓⠂⠀⠀⠀⠀⣀⣸⡥⠄⣏⠈⠙⠃⠀⠀⠀⠀⠀⠀⠀⠉⠓⠀⣀⣀⠀⠀⠀⠀⠀⠀⢀⡿⠀⡀⠀⣿⠀⠀⠀⠀
⠀⠀⣤⣤⣴⣦⡄⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⣤⣤⣤⡞⠒⠂⣿⠉⠉⠀⠀⠀⠀⠀⣤⣤⣴⠒⣿⠉⠉⠉⠀⠀⠀⠀⠀⠀⡞⠀⢠⣿⣤⣿⡖⠂⠀⠀
⠀⠀⢸⣅⣠⣏⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡗⠲⢦⡄⠀⠀⠀⠀⠁⠠⣼⣧⠖⠂⢸⡀⡷⠂⠀⠀⠀⠀⠀⠲⢄⠀⡿⠀⠀⠀⠀⠀⠀⠀⢀⠞⢳⡀⠊⢀⣀⣿⣀⣠⡤⠄
⠀⠀⠈⠃⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠺⠗⣿⣒⣀⢀⣿⠁⠀⠀⠀⠀⠀⠀⠀⢈⣿⣇⡀⠀⠀⠀⠀⠀⠀⠁⠀⢸⠈⠉⠉⠁⣿⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢤⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠐⠒⠋⣿⢁⠔⠋⠈⢷⣀⣈⠀⠀⠠⠤⠖⠋⠀⠈⠻⢶⣤⣀⣀⡀⠀⠀⠀⣿⠀⠀⠀⠀⢻⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠙⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⢉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⢤⣀⠀⣀⣠⣤⣤⡀⠀⠀⢀⡠⢤⣄⠀⠀⣀⣤⣤⣄⠀⣠⣄⠀⠀⣠⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠀⠀⠀⠀⢰⡏⢠⡟⠀⣹⣿⡁⢸⡇⢀⡿⠀⣰⠏⠀⠀⢻⡆⣟⠀⣿⢀⣼⠃⠁⠸⣇⡞⠁⠀⠀⠀⠀⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠀⠀⠀⠀⠈⠁⢸⡟⠋⠁⠈⠀⣸⠋⢿⡄⠀⣿⡀⠀⠀⣼⠃⠁⢀⣿⠉⢻⡆⠀⠀⣿⠀⠀⠀⠀⠀⠀⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠴⠿⠇⠀⠀⠀⡶⠿⠂⠈⢻⡞⠙⣷⣤⠞⠁⠀⠲⠾⠷⠶⠋⠀⠰⠾⠷⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

zhuzhu_text = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⡿⠋⠻⣿⣿⣄⠀⠀⣠⣤⣶⣦⣤⣶⣶⣶⣶⣤⣤⣤⣤⡀⠀⠀⠀⠀⠀⠀⢀⣰⣿⣿⣦⡀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⢠⣿⡿⠃⠀⠀⠈⠻⣿⣷⣼⣿⣿⠿⠿⠿⠛⠛⠛⠿⠿⠿⢿⣿⣿⣿⣦⣄⡀⣠⣾⣿⠟⠋⢻⣿⣧⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⢀⣾⣿⠇⠀⠀⠀⠀⠀⠘⠋⠉⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠿⣿⣿⣿⣿⡁⠀⠀⠀⣿⣿⡄⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⢸⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⡿⠛⠛⠁⠀⠀⠀⢸⣿⡧⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⣸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡀⠉⠃⠀⠀⠀
        ⠀⠀⠀⠀⠀⢠⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣿⡆⠀⠀⠀⠀
        ⠀⠀⠀⠀⣠⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⣤⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⡄⠀⠀⠀
        ⠀⠀⢀⣴⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⢀⣶⣿⣶⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣿⠀⠀⠀
        ⠀⠀⢸⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⠟⠁⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣧⠀⠀
        ⠀⢠⣿⣿⠃⠀⣼⣿⣦⣾⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⣴⡀⠀⠈⢿⣿⣧⠀
        ⠀⢸⣿⣿⠀⠰⠟⠁⡿⠋⠀⠀⠀⠀⠀⢀⣤⣶⣤⣴⣶⣶⣦⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢾⣿⣆⠻⣿⣦⠀⠈⣿⣿⡄
        ⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⠿⣿⡿⠛⣿⣹⣻⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠟⠀⠈⠁⠀⠀⣿⣿⠇
        ⠀⠈⢻⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣯⡀⠀⠀⢰⣿⡏⣼⣿⣿⠋⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀
        ⠀⠀⠘⠿⣿⣧⣄⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣦⣄⣘⣿⣀⣹⣿⣷⣶⣾⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡇⠀
        ⠀⠀⠀⠀⠹⢿⣿⣿⣦⣤⣀⡀⠀⠀⠀⠀⠈⠻⢿⣿⣿⣿⣿⠿⠿⠿⠛⠛⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⠇⠀
        ⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⠃⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣇⣀⣀⣀⣀⣴⣾⣿⡿⠃⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡿⠻⠿⠛⠛⠛⠿⠋⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⣠⣶⣿⣿⡿⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⢀⣴⣿⣿⠟⠋⠉⣀⣠⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⢠⣾⣿⠟⠁⣀⣴⣾⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢻⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀
        ⠻⣿⣿⣿⣿⣿⡿⠟⠉⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠀⠹⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀
        ⠀⠈⠉⠉⠉⠀⠀⠀⠀⢸⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡿⣿⣦⣤⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⣷⡀⠈⠹⣿⠛⠁⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⡷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⣠⣴⣿⣿⣿⡟⠀⠀⠀⣀⣠⣤⣴⣶⣤⣤⣄⡀⣀⣀⠀⠀⠀⠀⠀⠀⢰⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⢀⣾⣿⣿⠏⠉⢁⣀⣠⣶⣿⣿⣿⣿⡿⠿⢿⣿⣿⣿⣿⣿⣷⣤⡀⠀⠀⢠⣿⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠸⣿⣿⣷⣾⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠈⠛⠿⠿⣿⣿⣿⣿⠄⠀⠀⠛⢿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠉⠛⠛⠛⠋⠉⠙⠋⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⣿⣿⣷⣤⣀⢀⣀⣻⣿⣿⡦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⢿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢉⣙⣋⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""


def plot_zhuzhu():
    for line in prompt_text.strip().split("\n"):
        shared_logger.log(line)
    for line in zhuzhu_text.strip().split("\n"):
        time.sleep(0.5)
        shared_logger.log(line)
