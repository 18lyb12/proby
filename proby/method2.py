import chemprop
import time
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display

from proby.shared_logger import shared_logger
from pathlib import Path
import shutil
from proby.util import get_smiles, load_data

# Get the absolute path of the directory containing the current file
current_file_path = Path(__file__).resolve()
current_folder_path = current_file_path.parent

# model
model_15_dir = os.path.join(current_folder_path, 'models/model_1.5')
model_2_dir = os.path.join(current_folder_path, 'models/model_2')

# common
common_data_folder = os.path.join(current_folder_path, "data/common")

## model 1.5
reported_smiles_signal_path = os.path.join(common_data_folder, 'reported_smiles_signal.csv')

## model 2
common_solvents_path = os.path.join(common_data_folder, 'common_solvents.json')
scale_parameters_path = os.path.join(common_data_folder, 'scale_parameters.json')
reported_active_smiles_properties_path = os.path.join(common_data_folder, 'reported_active_smiles_properties.csv')

# input
# input_data_folder = os.path.join(current_folder_path, "data/input")

# smiles inventory
smiles_inventory_folder = os.path.join(current_folder_path, "data/smiles_inventory")

# intermediate
intermediate_data_folder = os.path.join(current_folder_path, "data/intermediate")

## model 1.5
model_15_data_path = os.path.join(intermediate_data_folder, 'model_1.5_data.csv')

model_15_single_data_path = os.path.join(intermediate_data_folder, "model_1.5_single_data.csv")

## model 2
model_2_data_path = os.path.join(intermediate_data_folder, 'model_2_data.csv')
model_2_scaled_preds_path = os.path.join(intermediate_data_folder, 'model_2_scaled_preds.csv')

model_2_single_data_path = os.path.join(intermediate_data_folder, "model_2_single_data.csv")
model_2_scaled_single_preds = os.path.join(intermediate_data_folder, 'model_2_scaled_single_preds.csv')

# output
output_data_folder = os.path.join(current_folder_path, "data/output2")

## model 1.5
model_15_preds_path = os.path.join(output_data_folder, 'model_1.5_preds.csv')
model_15_single_preds_path = os.path.join(output_data_folder, 'model_1.5_single_preds.csv')
model_15_single_interpret_path = os.path.join(output_data_folder, 'model_1.5_single_interpret.csv')

## model 2
model_2_preds_path = os.path.join(output_data_folder, 'model_2_preds.csv')
model_2_single_preds_path = os.path.join(output_data_folder, 'model_2_single_preds.csv')

# comprehensive prediction
comprehensive_folder = os.path.join(output_data_folder, "comprehensive")

# report
report_path = os.path.join(output_data_folder, "report", "report.csv")


def model_15(metadata):
    ### create smiles
    smiles_list = metadata["smiles_list"]
    smiles_df = pd.DataFrame({"smiles": smiles_list})

    model_15_data = smiles_df
    model_15_data.to_csv(model_15_data_path, index=False, encoding='utf-8-sig')

    ## predict model 1.5

    model_15_preds_df = predict_model_15(test_path=model_15_data_path,
                                         preds_path=model_15_preds_path)
    metadata["model_15_preds_df"] = model_15_preds_df

    df = model_15_preds_df[model_15_preds_df['new_category'] != "Invalid SMILES"]
    df['new_category'] = df['new_category'].astype(float)

    ### load reported smiles signal
    reported_smiles_signal_df = pd.read_csv(reported_smiles_signal_path)
    metadata["reported_smiles_signal_df"] = reported_smiles_signal_df

    df = pd.merge(df, reported_smiles_signal_df, on='smiles', how='left')

    ## pred 1.5 smiles
    # DIY
    threshold = 0.95
    metadata["threshold"] = threshold
    df['new_category'] = df['new_category'].astype(float)
    df['true_category'] = df['true_category'].astype(float)
    df["high_pred_score"] = df['new_category'].apply(lambda x: 1 if x >= threshold else 0)
    # preds_15 = df[(df['high_pred_score'] == 1) | (df['true_category'] == 1)]
    preds_15 = df[((df['true_category'].isna()) & (df['high_pred_score'] == 1)) | (df['true_category'] == 1)]
    preds_15.head(5)

    ## selected_smiles
    selected_smiles = preds_15["smiles"].to_list()
    shared_logger.log(f"{len(selected_smiles)} selected smiles in total, including {', '.join(selected_smiles[:5])}, etc.")
    metadata["selected_smiles"] = selected_smiles


def model_2(metadata):
    ### create selected_smiles
    selected_smiles = metadata["selected_smiles"]
    selected_smiles_df = pd.DataFrame({"Smiles": selected_smiles})

    ### load common_solvents
    with open(common_solvents_path, 'r') as file:
        common_solvents = json.load(file)
    metadata["common_solvents"] = common_solvents
    solvents_df = pd.DataFrame(common_solvents, columns=["Solvent"])

    ### cross join
    selected_smiles_df['key'] = 1
    solvents_df['key'] = 1
    model_2_data = pd.merge(selected_smiles_df, solvents_df, on='key', how='outer').drop('key', axis=1)
    model_2_data.to_csv(model_2_data_path, index=False, encoding='utf-8-sig')

    ## predict model 2
    model_2_preds_df = predict_model_2(test_path=model_2_data_path,
                                       scaled_preds_path=model_2_scaled_preds_path,
                                       preds_path=model_2_preds_path)
    metadata["model_2_preds_df"] = model_2_preds_df

    ## load reported active smiles properties path
    reported_active_smiles_properties_df = pd.read_csv(reported_active_smiles_properties_path)
    reported_active_smiles_properties_df["properties reported"] = 1
    metadata["reported_active_smiles_properties_df"] = reported_active_smiles_properties_df

def generate_comprehensive_prediction(metadata):
    # Generate Comprehensive Prediction
    model_15_preds_df = metadata["model_15_preds_df"]
    threshold = metadata["threshold"]
    model_15_preds_df["high_pred_score"] = model_15_preds_df['new_category'].apply(lambda x: 1 if (x != "Invalid SMILES" and float(x) >= threshold) else 0)
    model_15_preds_df.rename(columns={"new_category": "activity_score"}, inplace=True)

    reported_smiles_signal_df = metadata["reported_smiles_signal_df"]
    model_15_preds_report_df = pd.merge(model_15_preds_df, reported_smiles_signal_df, on='smiles', how='left')
    model_15_preds_report_df['true_category'] = model_15_preds_report_df['true_category'].fillna("NA")
    model_15_preds_report_df.rename(columns={"smiles": "smiles_"}, inplace=True)

    input_data_folder = metadata["input_data_folder"]
    model_2_preds_df = metadata["model_2_preds_df"]
    reported_active_smiles_properties_df = metadata["reported_active_smiles_properties_df"]
    common_solvents = metadata["common_solvents"]
    app_output_data_folder = metadata["app_output_data_folder"]
    for file_name in os.listdir(input_data_folder):
        if file_name.endswith('.xlsx'):
            full_path = os.path.join(input_data_folder, file_name)
            df = pd.read_excel(full_path, dtype=str)
            column_name = [column for column in df.columns if column.lower() == "smiles"][0]
            df.rename(columns={column_name: "SMILES"}, inplace=True)
            column_name = "SMILES"
            # col_idx = df.columns.get_loc(column_name)
            # col_letter = get_column_letter(col_idx)

            df[column_name] = df[column_name].apply(lambda x: x.strip() if isinstance(x, str) else x)

            output_full_path = os.path.join(comprehensive_folder, file_name)
            print(f"generating {output_full_path} ...")
            with pd.ExcelWriter(output_full_path) as writer:
                for i, solvent in enumerate(common_solvents):
                    print(f"\tgenerating sheet {solvent} ...")

                    merged_preds_report_15 = pd.merge(df, model_15_preds_report_df, left_on=column_name, right_on='smiles_', how='left')

                    merged_preds_2 = pd.merge(merged_preds_report_15, model_2_preds_df[model_2_preds_df['Solvent'] == solvent], left_on='smiles_', right_on='Smiles', how='left')
                    merged_preds_2["Solvent"] = solvent

                    comprehensive_preds = pd.merge(merged_preds_2, reported_active_smiles_properties_df, on=["Smiles", "Solvent"], how='left')
                    comprehensive_preds['properties reported'] = comprehensive_preds['properties reported'].fillna(0)
                    comprehensive_preds = comprehensive_preds.drop(columns=["smiles_", "Smiles", "Solvent"])

                    sheet_name = f"{solvent} ({i + 1})"
                    comprehensive_preds.to_excel(writer, sheet_name=sheet_name, index=False)

                    # workbook  = writer.book
                    # worksheet = writer.sheets[sheet_name]
                    # # set the hyperlink
                    # for idx, row in comprehensive_preds.iterrows():
                    #     smiles = row[column_name]
                    #     img_path = os.path.join(smiles_inventory_folder, smiles2path[smiles])
                    #     hyperlink = os.path.join("../../..", smiles_inventory_folder, smiles2path[smiles])
                    #     if os.path.isfile(img_path):
                    #         cell = f'{col_letter}{idx + 2}'
                    #         worksheet[cell].hyperlink = hyperlink
            shutil.copy(output_full_path, os.path.join(app_output_data_folder, f"method 2 processed {file_name}"))

def generate_report(metadata):
    all_input_smiles = set()
    all_model_15_reported_smiles = set()
    all_model_15_reported_positive_smiles = set()
    all_model_15_reported_negative_smiles = set()
    all_model_15_not_reported_smiles = set()
    all_model_15_not_reported_positive_smiles = set()
    all_model_15_not_reported_negative_smiles = set()
    all_model_2_candidates_pairs = set()
    all_model_2_property_reported_pairs = set()
    all_model_2_property_not_reported_pairs = set()
    report = {"file_name": [],
              "input_smiles": [],
              "model_1.5_reported_smiles": [],
              "model_1.5_reported_positive_smiles": [],
              "model_1.5_reported_negative_smiles": [],
              "model_1.5_not_reported_smiles": [],
              "model_1.5_not_reported_positive_smiles": [],
              "model_1.5_not_reported_negative_smiles": [],
              "model_2_candidates_pairs": [],
              "model_2_property_reported_pairs": [],
              "model_2_property_not_reported_pairs": [],
              }
    for file_name in os.listdir(comprehensive_folder):
        print(f"{file_name}")

        total_input_smiles = set()
        total_model_15_reported_smiles = set()
        total_model_15_reported_positive_smiles = set()
        total_model_15_reported_negative_smiles = set()
        total_model_15_not_reported_smiles = set()
        total_model_15_not_reported_positive_smiles = set()
        total_model_15_not_reported_negative_smiles = set()
        total_model_2_candidates_pairs = set()
        total_model_2_property_reported_pairs = set()
        total_model_2_property_not_reported_pairs = set()

        if file_name.endswith('.xlsx'):
            full_path = os.path.join(comprehensive_folder, file_name)
            excel_file = pd.ExcelFile(full_path, engine='openpyxl')

            for sheet_name in excel_file.sheet_names:
                print(f"  {sheet_name}")
                solvent = sheet_name.split()[0]
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                df = df[df["activity_score"] != "Invalid SMILES"]
                column_name = [column for column in df.columns if column.lower() == "smiles"][0]
                input_smiles = get_smiles(df, column_name)

                model_15_reported_df = df[~df["true_category"].isna()]
                model_15_reported_positive_df = model_15_reported_df[model_15_reported_df["true_category"] == 1]
                model_15_reported_negative_df = model_15_reported_df[model_15_reported_df["true_category"] == 0]

                model_15_reported_smiles = get_smiles(model_15_reported_df, column_name)
                model_15_reported_positive_smiles = get_smiles(model_15_reported_positive_df, column_name)
                model_15_reported_negative_smiles = get_smiles(model_15_reported_negative_df, column_name)

                model_15_not_reported_df = df[df["true_category"].isna()]
                model_15_not_reported_positive_df = model_15_not_reported_df[model_15_not_reported_df["high_pred_score"] == 1]
                model_15_not_reported_negative_df = model_15_not_reported_df[model_15_not_reported_df["high_pred_score"] == 0]

                model_15_not_reported_smiles = get_smiles(model_15_not_reported_df, column_name)
                model_15_not_reported_positive_smiles = get_smiles(model_15_not_reported_positive_df, column_name)
                model_15_not_reported_negative_smiles = get_smiles(model_15_not_reported_negative_df, column_name)

                model_2_candidates_df = df[(df["true_category"] == 1) | ((df["true_category"].isna()) & (df["high_pred_score"] == 1))]
                model_2_property_reported_df = model_2_candidates_df[model_2_candidates_df["properties reported"] == 1]
                model_2_property_not_reported_df = model_2_candidates_df[model_2_candidates_df["properties reported"] == 0]

                model_2_candidates_pairs = {(smiles, solvent) for smiles in get_smiles(model_2_candidates_df, column_name)}
                model_2_property_reported_pairs = {(smiles, solvent) for smiles in get_smiles(model_2_property_reported_df, column_name)}
                model_2_property_not_reported_pairs = {(smiles, solvent) for smiles in get_smiles(model_2_property_not_reported_df, column_name)}

                print(f"\t there are {len(input_smiles)} smiles in input data.")

                print(f"\t\t {len(model_15_reported_smiles)} are reported. "
                      f"{len(model_15_reported_positive_smiles)} positive, "
                      f"{len(model_15_reported_negative_smiles)} negative")

                print(f"\t\t {len(model_15_not_reported_smiles)} are not reported. "
                      f"{len(model_15_not_reported_positive_smiles)} positive, "
                      f"{len(model_15_not_reported_negative_smiles)} negative")

                print(f"\t there are {len(model_2_candidates_pairs)} (smiles, solvent) pairs are predicted by model 2.")
                print(f"\t\t {len(model_2_property_reported_pairs)} are reported,")
                print(f"\t\t {len(model_2_property_not_reported_pairs)} are not reported.")

                assert len(model_2_candidates_pairs) == len(model_15_reported_positive_smiles) + len(model_15_not_reported_positive_smiles)
                total_input_smiles |= input_smiles
                total_model_15_reported_smiles |= model_15_reported_smiles
                total_model_15_reported_positive_smiles |= model_15_reported_positive_smiles
                total_model_15_reported_negative_smiles |= model_15_reported_negative_smiles
                total_model_15_not_reported_smiles |= model_15_not_reported_smiles
                total_model_15_not_reported_positive_smiles |= model_15_not_reported_positive_smiles
                total_model_15_not_reported_negative_smiles |= model_15_not_reported_negative_smiles
                total_model_2_candidates_pairs |= model_2_candidates_pairs
                total_model_2_property_reported_pairs |= model_2_property_reported_pairs
                total_model_2_property_not_reported_pairs |= model_2_property_not_reported_pairs

        print(f" there are {len(total_input_smiles)} smiles in input data.")

        print(f"\t {len(total_model_15_reported_smiles)} are reported. "
              f"{len(total_model_15_reported_positive_smiles)} positive, "
              f"{len(total_model_15_reported_negative_smiles)} negative")

        print(f"\t {len(total_model_15_not_reported_smiles)} are not reported. "
              f"{len(total_model_15_not_reported_positive_smiles)} positive, "
              f"{len(total_model_15_not_reported_negative_smiles)} negative")

        print(f" there are {len(total_model_2_candidates_pairs)} (smiles, solvent) pairs are predicted by model 2.")
        print(f"\t {len(total_model_2_property_reported_pairs)} are reported,")
        print(f"\t {len(total_model_2_property_not_reported_pairs)} are not reported.")

        report["file_name"].append(file_name)
        report["input_smiles"].append(len(total_input_smiles))
        report["model_1.5_reported_smiles"].append(len(total_model_15_reported_smiles))
        report["model_1.5_reported_positive_smiles"].append(len(total_model_15_reported_positive_smiles))
        report["model_1.5_reported_negative_smiles"].append(len(total_model_15_reported_negative_smiles))
        report["model_1.5_not_reported_smiles"].append(len(total_model_15_not_reported_smiles))
        report["model_1.5_not_reported_positive_smiles"].append(len(total_model_15_not_reported_positive_smiles))
        report["model_1.5_not_reported_negative_smiles"].append(len(total_model_15_not_reported_negative_smiles))
        report["model_2_candidates_pairs"].append(len(total_model_2_candidates_pairs))
        report["model_2_property_reported_pairs"].append(len(total_model_2_property_reported_pairs))
        report["model_2_property_not_reported_pairs"].append(len(total_model_2_property_not_reported_pairs))

        all_input_smiles |= total_input_smiles
        all_model_15_reported_smiles |= total_model_15_reported_smiles
        all_model_15_reported_positive_smiles |= total_model_15_reported_positive_smiles
        all_model_15_reported_negative_smiles |= total_model_15_reported_negative_smiles
        all_model_15_not_reported_smiles |= total_model_15_not_reported_smiles
        all_model_15_not_reported_positive_smiles |= total_model_15_not_reported_positive_smiles
        all_model_15_not_reported_negative_smiles |= total_model_15_not_reported_negative_smiles
        all_model_2_candidates_pairs |= total_model_2_candidates_pairs
        all_model_2_property_reported_pairs |= total_model_2_property_reported_pairs
        all_model_2_property_not_reported_pairs |= total_model_2_property_not_reported_pairs


    print(f"there are {len(all_input_smiles)} smiles in input data.")

    print(f" {len(all_model_15_reported_smiles)} are reported. "
          f"{len(all_model_15_reported_positive_smiles)} positive, "
          f"{len(all_model_15_reported_negative_smiles)} negative")

    print(f" {len(all_model_15_not_reported_smiles)} are not reported. "
          f"{len(all_model_15_not_reported_positive_smiles)} positive, "
          f"{len(all_model_15_not_reported_negative_smiles)} negative")

    print(f"there are {len(all_model_2_candidates_pairs)} (smiles, solvent) pairs are predicted by model 2.")
    print(f" {len(all_model_2_property_reported_pairs)} are reported,")
    print(f" {len(all_model_2_property_not_reported_pairs)} are not reported.")

    report["file_name"].append("all")
    report["input_smiles"].append(len(all_input_smiles))
    report["model_1.5_reported_smiles"].append(len(all_model_15_reported_smiles))
    report["model_1.5_reported_positive_smiles"].append(len(all_model_15_reported_positive_smiles))
    report["model_1.5_reported_negative_smiles"].append(len(all_model_15_reported_negative_smiles))
    report["model_1.5_not_reported_smiles"].append(len(all_model_15_not_reported_smiles))
    report["model_1.5_not_reported_positive_smiles"].append(len(all_model_15_not_reported_positive_smiles))
    report["model_1.5_not_reported_negative_smiles"].append(len(all_model_15_not_reported_negative_smiles))
    report["model_2_candidates_pairs"].append(len(all_model_2_candidates_pairs))
    report["model_2_property_reported_pairs"].append(len(all_model_2_property_reported_pairs))
    report["model_2_property_not_reported_pairs"].append(len(all_model_2_property_not_reported_pairs))

    report_df = pd.DataFrame(data=report)

    report_df.to_csv(report_path, index=False)

    app_output_data_folder = metadata["app_output_data_folder"]
    shutil.copy(report_path, os.path.join(app_output_data_folder, "method 2 report.csv"))


def method2(metadata):
    load_data(metadata)
    model_15(metadata)
    model_2(metadata)
    generate_comprehensive_prediction(metadata)
    generate_report(metadata)


def predict_model_15(test_path, preds_path):
    arguments = [
        '--test_path', test_path,
        '--preds_path', preds_path,
        '--checkpoint_dir', model_15_dir,
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    t0 = time.time()
    preds = chemprop.train.make_predictions(args=args)
    t1 = time.time()
    print(f"predicting time: {t1 - t0} s")
    df = pd.read_csv(preds_path)
    return df


def predict_model_2(test_path, scaled_preds_path, preds_path):
    arguments = [
        '--test_path', test_path,
        '--preds_path', scaled_preds_path,
        '--checkpoint_dir', model_2_dir,
        '--number_of_molecules', '2',
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)

    t0 = time.time()
    preds = chemprop.train.make_predictions(args=args)
    t1 = time.time()
    print(f"predicting time: {t1 - t0} s")

    df = pd.read_csv(scaled_preds_path)
    df = df[~df.apply(lambda row: row.eq('Invalid SMILES').any(), axis=1)]

    with open(scale_parameters_path, 'r') as json_file:
        scale_parameters = json.load(json_file)

    target_columns = ["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)", "Quantum yield", "log(e/mol-1 dm3 cm-1)", "abs FWHM (cm-1)", "emi FWHM (cm-1)", "abs FWHM (nm)", "emi FWHM (nm)"]
    for target_column in target_columns:
        mean = scale_parameters[target_column]["mean"]
        std_dev = scale_parameters[target_column]["std_dev"]
        df[target_column] = df[f"Scaled {target_column}"].apply(lambda x: max(0, x * std_dev + mean) if pd.notna(x) and x != '' else x)
        df = df.drop(f"Scaled {target_column}", axis=1)
    df.to_csv(preds_path, index=False, encoding='utf-8-sig')
    return df


def interpret_model_15(smiles, prop_delta=0.99):
    model_15_single_data = pd.DataFrame({"smiles": [smiles], "new_category": [1]})
    model_15_single_data.to_csv(model_15_single_data_path, index=False, encoding='utf-8-sig')

    arguments = [
        '--data_path', model_15_single_data_path,
        '--smiles_columns', 'smiles',
        '--property_id', '1',
        '--checkpoint_dir', model_15_dir,
        '--interpret_output_path', model_15_single_interpret_path,
        '--prop_delta', str(prop_delta)
    ]

    args = chemprop.args.InterpretArgs().parse_args(arguments)
    preds = chemprop.interpret.interpret(args=args)

    original_smiles = preds[0][0]
    sub_smiles = preds[0][2]
    rationale_score = preds[0][3]

    return original_smiles, sub_smiles, rationale_score