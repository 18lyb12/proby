import json
import matplotlib.pyplot as plt
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


# process `chemfluo的数据集` which has both positive and negative data
def process_chemfluo_data():
    print("start to process chemfluo的数据集")

    chemfluo_data_folder = os.path.join(raw_data_folder, 'chemfluo的数据集')
    threshold = 20
    list_of_dfs = []
    absorption_emission_pairs = []
    for f in os.scandir(chemfluo_data_folder):
        if f.is_dir():
            sub_folder = f.name
            for file_name in os.listdir(os.path.join(chemfluo_data_folder, sub_folder)):
                if file_name.endswith('.csv'):
                    full_path = os.path.join(chemfluo_data_folder, sub_folder, file_name)
                    print(f"processing {full_path}...")
                    df = pd.read_csv(full_path, dtype=str)
                    # row number of first useful row
                    row_number = (df[df.columns[0]] == "1").idxmax()
                    df = pd.read_csv(full_path, skiprows=row_number, usecols=[0, 3, 4, 5])
                    df.columns = ["id", "smiles", "category", "score"]
                    df['category'] = df['category'].str.lower()
                    absorption_max, emission_max = file_name.split(".csv")[0].split(" ")[-2:]
                    absorption_emission_pairs += [(float(absorption_max), float(emission_max))] * len(df)
                    df["absorption_max"] = int(absorption_max)
                    df["emission_max"] = int(emission_max)
                    df['new_category'] = df['score'].apply(lambda x: 1 if x > threshold else 0)
                    df["full_path"] = full_path
                    # some analysis
                    inactive_max = df[df["category"] == "inactive"]['score'].max()
                    active_min = df[df["category"] == "active"]['score'].min()
                    print(inactive_max, active_min)
                    list_of_dfs.append(df)

    # this is model 1 data
    chemfluo_data_df = pd.concat(list_of_dfs)

    print("finish processing chemfluo的数据集")

    return {"chemfluo_data_df": chemfluo_data_df,
            "absorption_emission_pairs": absorption_emission_pairs}


# process `下载数据+人工整理.xlsx` which are all positive data and have smiles properties information
def process_download_human_data(processed_data):
    print("start to process 下载数据+人工整理.xlsx")
    chemfluo_data_df = processed_data["chemfluo_data_df"]
    absorption_emission_pairs = processed_data["absorption_emission_pairs"]
    file_path = os.path.join(raw_data_folder, "下载数据+人工整理.xlsx")
    xls = pd.ExcelFile(file_path)
    download_human_data_dfs = []
    for sheet_name in xls.sheet_names:  # 遍历
        print(f"processing {sheet_name} ...")
        download_human_data_df = pd.read_excel(xls, sheet_name)
        download_human_data_df["id"] = download_human_data_df.index
        download_human_data_df["smiles"] = download_human_data_df["Smiles"]
        download_human_data_df[["absorption_max", "emission_max"]] = download_human_data_df[
            ["Absorption max (nm)", "Emission max (nm)"]]
        for _, row in download_human_data_df.iterrows():
            absorption_emission_pairs.append((float(row["absorption_max"]), float(row["emission_max"])))
        download_human_data_df["category"] = "active"
        download_human_data_df["score"] = 100
        download_human_data_df["new_category"] = 1
        download_human_data_df["full_path"] = os.path.join(file_path, sheet_name)
        download_human_data_df = download_human_data_df[chemfluo_data_df.columns]
        download_human_data_dfs.append(download_human_data_df)
        plt.scatter(download_human_data_df['absorption_max'], download_human_data_df['emission_max'], alpha=0.1,
                    label=sheet_name)

    with open(os.path.join(common_data_folder, "common_absorption_emission_pairs_temp.json"), 'w') as json_file:
        common_absorption_emission_pairs = [pair for pair, freq in Counter(absorption_emission_pairs).most_common() if
                                            freq >= 100]
        json.dump(common_absorption_emission_pairs, json_file, indent=4)

    plt.scatter(chemfluo_data_df['absorption_max'], chemfluo_data_df['emission_max'], alpha=0.1, label='chemfluo_data')
    plt.xlabel('absorption_max')
    plt.ylabel('emission_max')
    plt.title(f'absorption_max-emission_max Graph')
    plt.legend()
    plt.grid(True)
    image_file_name = 'absorption_max-emission_max distribution.png'
    image_path = os.path.join(intermediate_data_folder, image_file_name)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"{image_file_name} is saved at {image_path}")

    concatenated_df = pd.concat(download_human_data_dfs + [chemfluo_data_df])
    concatenated_df.dropna(subset=['smiles', 'absorption_max', 'emission_max'], inplace=True)
    concatenated_df = concatenated_df.drop_duplicates(subset=['smiles', 'absorption_max', 'emission_max'])

    print("finish processing 下载数据+人工整理.xlsx")
    processed_data["concatenated_df"] = concatenated_df
    return processed_data


def group_by_smiles(df):
    df["absorption_max_category"] = df["absorption_max"].apply(lambda x: int(float(x) / 100))
    df["emission_max_category"] = df["emission_max"].apply(lambda x: int(float(x) / 100))

    def custom_agg(x):
        return 1 if len(set([_ for _ in x if 3 <= _ <= 6])) >= 3 else 0

    # grouped_df = df.groupby('smiles')['new_category'].max().reset_index()
    grouped_df = df.groupby('smiles').agg({'new_category': 'max',
                                           'absorption_max_category': custom_agg,
                                           'emission_max_category': custom_agg
                                           }).reset_index()
    grouped_df = grouped_df[(grouped_df['new_category'] == 1) | (grouped_df['absorption_max_category'] == 1) | (
                grouped_df['emission_max_category'] == 1)]
    grouped_df["id"] = grouped_df.index
    grouped_df = grouped_df.loc[:, ["id", "smiles", "new_category"]]
    return grouped_df


def print_positive_negative(df):
    negative = df[df['new_category'] == 0]
    positive = df[df['new_category'] == 1]
    print_data_distribution(positive, "positive", negative, "negative", df)


def split_data(processed_data):
    print("start to split data for model 1")
    # smiles在train_val和test里不重复
    concatenated_df = processed_data["concatenated_df"]

    smiles_set = set(concatenated_df['smiles'])

    random.seed(42)
    test_smiles_set = set(random.sample(list(smiles_set), int(0.1 * len(smiles_set))))
    train_val_smiles_set = smiles_set.difference(test_smiles_set)

    print_data_distribution(train_val_smiles_set, "train_val_smiles_set", test_smiles_set, "test_smiles_set",
                            smiles_set)

    test_df = concatenated_df[concatenated_df['smiles'].isin(test_smiles_set)]
    train_val_df = concatenated_df[concatenated_df['smiles'].isin(train_val_smiles_set)]
    print_data_distribution(train_val_df, "train_val_df", test_df, "test_df", concatenated_df)

    test_df_negative = test_df[test_df['new_category'] == 0]
    test_df_positive = test_df[test_df['new_category'] == 1]
    print_data_distribution(test_df_positive, "test_df_positive", test_df_negative, "test_df_negative", test_df)

    train_val_df_negative = train_val_df[train_val_df['new_category'] == 0]
    train_val_df_positive = train_val_df[train_val_df['new_category'] == 1]
    print_data_distribution(train_val_df_positive, "train_val_df_positive", train_val_df_negative,
                            "train_val_df_negative", train_val_df)

    processed_data["train_val_df"] = train_val_df
    processed_data["test_df"] = test_df

    train_val_df.to_csv(os.path.join(processed_data_folder, "model_1_train_val_data.csv"), index=False,
                        encoding='utf-8-sig')
    train_val_features_df = train_val_df[["absorption_max", "emission_max"]]
    train_val_features_df.to_csv(os.path.join(processed_data_folder, "model_1_train_val_features.csv"), index=False,
                                 encoding='utf-8-sig')

    test_df.to_csv(os.path.join(processed_data_folder, "model_1_test_full_data.csv"), index=False, encoding='utf-8-sig')
    test_df[["smiles"]].to_csv(os.path.join(processed_data_folder, "model_1_test_data.csv"), index=False,
                               encoding='utf-8-sig')
    test_features_df = test_df[["absorption_max", "emission_max"]]
    test_features_df.to_csv(os.path.join(processed_data_folder, "model_1_test_features.csv"), index=False,
                            encoding='utf-8-sig')

    print("finish splitting data for model 1")
    return processed_data


def process_model_15_data(processed_data):
    print("start to process model 1.5 data")
    train_val_df = processed_data["train_val_df"]
    test_df = processed_data["test_df"]
    grouped_test_df = group_by_smiles(test_df)
    grouped_train_val_df = group_by_smiles(train_val_df)
    grouped_train_val_df_negative = grouped_train_val_df[grouped_train_val_df['new_category'] == 0]
    grouped_train_val_df_positive = grouped_train_val_df[grouped_train_val_df['new_category'] == 1]
    print_data_distribution(grouped_train_val_df_positive, "grouped_train_val_df_positive",
                            grouped_train_val_df_negative, "grouped_train_val_df_negative", grouped_train_val_df)
    grouped_test_df_negative = grouped_test_df[grouped_test_df['new_category'] == 0]
    grouped_test_df_positive = grouped_test_df[grouped_test_df['new_category'] == 1]
    print_data_distribution(grouped_test_df_positive, "grouped_test_df_positive", grouped_test_df_negative,
                            "grouped_test_df_negative", grouped_test_df)

    grouped_train_val_df.to_csv(os.path.join(processed_data_folder, "model_1.5_train_val_data.csv"), index=False,
                                encoding='utf-8-sig')

    grouped_test_df.to_csv(os.path.join(processed_data_folder, "model_1.5_test_full_data.csv"), index=False,
                           encoding='utf-8-sig')
    grouped_test_df[["smiles"]].to_csv(os.path.join(processed_data_folder, "model_1.5_test_data.csv"), index=False,
                                       encoding='utf-8-sig')

    print("finish processing model 1.5 data")


def main():
    processed_data = process_chemfluo_data()
    processed_data = process_download_human_data(processed_data)
    concatenated_df = processed_data["concatenated_df"]
    grouped_concatenated_df = group_by_smiles(concatenated_df)
    print_positive_negative(concatenated_df)
    print_positive_negative(grouped_concatenated_df)

    # split data for model 1
    processed_data = split_data(processed_data)

    # generate data for model 1.5
    process_model_15_data(processed_data)


if __name__ == "__main__":
    main()
