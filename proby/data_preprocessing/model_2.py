import json
import os
import pandas as pd
import random
from collections import Counter
from pathlib import Path

from proby.data_preprocessing.util import print_data_distribution

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
raw_data_folder = os.path.join(root_folder_path, 'data/raw_data')
intermediate_data_folder = os.path.join(root_folder_path, 'data/intermediate')
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
common_data_folder = os.path.join(root_folder_path, "data/common")


# process `下载数据+人工整理.xlsx` which are all positive data and have smiles properties information
def process_download_human_data():
    print("start to process 下载数据+人工整理.xlsx")
    file_path = os.path.join(raw_data_folder, "下载数据+人工整理.xlsx")
    xls = pd.ExcelFile(file_path)
    download_human_data_dfs = []
    for sheet_name in xls.sheet_names:  # 遍历
        print(f"processing {sheet_name} ...")
        download_human_data_df = pd.read_excel(xls, sheet_name)
        download_human_data_dfs.append(download_human_data_df)

    df = pd.concat(download_human_data_dfs)
    mask = df['Emission max (nm)'] < df['Absorption max (nm)']
    df.loc[mask, ['Absorption max (nm)', 'Emission max (nm)']] = ''
    df.replace('NaN', '', inplace=True)
    print("finish processing 下载数据+人工整理.xlsx")
    return df


def split_data(df):
    print("start to split data for model 2")
    smiles_set = set(df['Smiles'])
    random.seed(42)
    test_smiles_set = set(random.sample(list(smiles_set), int(0.1 * len(smiles_set))))
    train_val_smiles_set = smiles_set.difference(test_smiles_set)
    print_data_distribution(train_val_smiles_set, "train_val_smiles_set", test_smiles_set, "test_smiles_set",
                            smiles_set)

    test_df = df[df['Smiles'].isin(test_smiles_set)]
    train_val_df = df[df['Smiles'].isin(train_val_smiles_set)]
    print_data_distribution(train_val_df, "train_val_df", test_df, "test_df", df)

    scale_parameters = {}
    for target_column in ["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)", "Quantum yield",
                          "log(e/mol-1 dm3 cm-1)", "abs FWHM (cm-1)", "emi FWHM (cm-1)", "abs FWHM (nm)",
                          "emi FWHM (nm)"]:
        values = train_val_df[target_column].dropna().loc[lambda x: x != ''].tolist()
        # mean, std_dev = np.mean(values), np.std(values)
        mean, std_dev = 0, 1
        print(target_column, mean, std_dev)
        scale_parameters[target_column] = {"mean": mean, "std_dev": std_dev}
        train_val_df[f"Scaled {target_column}"] = train_val_df[target_column].apply(
            lambda x: (x - mean) / std_dev if pd.notna(x) and x != '' else x)
        test_df[f"Scaled {target_column}"] = test_df[target_column].apply(
            lambda x: (x - mean) / std_dev if pd.notna(x) and x != '' else x)

    with open(os.path.join(common_data_folder, "scale_parameters_temp.json"), 'w') as json_file:
        json.dump(scale_parameters, json_file, indent=4)

    with open(os.path.join(common_data_folder, "common_solvents_temp.json"), 'w') as json_file:
        common_solvents = [solvent for solvent, freq in Counter(df["Solvent"]).most_common() if freq >= 100]
        json.dump(common_solvents, json_file, indent=4)

    train_val_df.to_csv(os.path.join(processed_data_folder, "model_2_train_val_data.csv"), index=False,
                        encoding='utf-8-sig')
    test_df.to_csv(os.path.join(processed_data_folder, "model_2_test_full_data.csv"), index=False, encoding='utf-8-sig')
    test_df[["Smiles", "Solvent"]].to_csv(os.path.join(processed_data_folder, "model_2_test_data.csv"), index=False,
                                          encoding='utf-8-sig')

    print("finish splitting data for model 2")


def main():
    df = process_download_human_data()
    split_data(df)


if __name__ == "__main__":
    main()
