import chemprop
import json
import os
import pandas as pd
import shutil
import threading
import time
from pathlib import Path

from proby.util import get_smiles, load_data, shared_logger, PredictionWithProgress

# Get the absolute path of the directory containing the current file
current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parent

# model
model_1_dir = os.path.join(root_folder_path, 'models/model_1')
model_2_dir = os.path.join(root_folder_path, 'models/model_2')

# common
common_data_folder = os.path.join(root_folder_path, "data/common")

## model 1
common_absorption_emission_pairs_path = os.path.join(common_data_folder, 'common_absorption_emission_pairs.json')
reported_smiles_signal_path = os.path.join(common_data_folder, 'reported_smiles_signal.csv')

## model 2
common_solvents_path = os.path.join(common_data_folder, 'common_solvents.json')
scale_parameters_path = os.path.join(common_data_folder, 'scale_parameters.json')
reported_active_smiles_properties_path = os.path.join(common_data_folder, 'reported_active_smiles_properties.csv')

# input
# input_data_folder = os.path.join(root_folder_path, "data/prediction_data/input")

# smiles inventory
smiles_inventory_folder = os.path.join(root_folder_path, "data/prediction_data/smiles_inventory")

# intermediate
intermediate_data_folder = os.path.join(root_folder_path, "data/prediction_data/intermediate")

## model 1
model_1_data_path = os.path.join(intermediate_data_folder, 'model_1_data.csv')
model_1_features_path = os.path.join(intermediate_data_folder, 'model_1_features.csv')

model_1_single_data_path = os.path.join(intermediate_data_folder, "model_1_single_data.csv")
model_1_single_features_path = os.path.join(intermediate_data_folder, "model_1_single_features.csv")

## model 2
model_2_data_path = os.path.join(intermediate_data_folder, 'model_2_data.csv')
model_2_scaled_preds_path = os.path.join(intermediate_data_folder, 'model_2_scaled_preds.csv')

model_2_single_data_path = os.path.join(intermediate_data_folder, "model_2_single_data.csv")
model_2_scaled_single_preds = os.path.join(intermediate_data_folder, 'model_2_scaled_single_preds.csv')

# output
output_data_folder = os.path.join(root_folder_path, "data/prediction_data/output1")

## model 1
model_1_preds_path = os.path.join(output_data_folder, 'model_1_preds.csv')
model_1_single_preds_path = os.path.join(output_data_folder, 'model_1_single_preds.csv')

## model 2
model_2_preds_path = os.path.join(output_data_folder, 'model_2_preds.csv')
model_2_single_preds_path = os.path.join(output_data_folder, 'model_2_single_preds.csv')

# comprehensive prediction
comprehensive_folder = os.path.join(output_data_folder, "comprehensive")

# report
report_path = os.path.join(output_data_folder, "report", "report.csv")


def model_1(metadata):
    shared_logger.log("model 1 session starts")
    ### create smiles
    shared_logger.log("creating smiles")
    smiles_list = metadata["smiles_list"]
    smiles_df = pd.DataFrame({"smiles": smiles_list})

    ### load common_absorption_emission_pairs
    shared_logger.log("loading common_absorption_emission_pairs")
    with open(common_absorption_emission_pairs_path, 'r') as file:
        common_absorption_emission_pairs = json.load(file)
    absorption_emission_df = pd.DataFrame(common_absorption_emission_pairs, columns=["absorption_max", "emission_max"])

    ### crose join
    smiles_df['key'] = 1
    absorption_emission_df['key'] = 1
    model_1_data = pd.merge(smiles_df, absorption_emission_df, on='key', how='outer').drop('key', axis=1)
    model_1_data.to_csv(model_1_data_path, index=False, encoding='utf-8-sig')
    model_1_data[["absorption_max", "emission_max"]].to_csv(model_1_features_path, index=False, encoding='utf-8-sig')

    ## predict model 1
    prediction_with_progress = PredictionWithProgress()
    wait_thread = threading.Thread(target=prediction_with_progress.print_wait_message, args=("predicting model 1",))
    wait_thread.start()
    model_1_preds_df = predict_model_1(test_path=model_1_data_path,
                                       features_path=model_1_features_path,
                                       preds_path=model_1_preds_path)
    prediction_with_progress.stop_flag.set()
    wait_thread.join()
    metadata["model_1_preds_df"] = model_1_preds_df

    df = model_1_preds_df[model_1_preds_df['new_category'] != "Invalid SMILES"]
    df['new_category'] = df['new_category'].astype(float)

    grouped_df = df.groupby('smiles')['new_category'].max().reset_index()

    ### load reported smiles signal
    shared_logger.log("loading reported smiles signal")
    reported_smiles_signal_df = pd.read_csv(reported_smiles_signal_path)
    metadata["reported_smiles_signal_df"] = reported_smiles_signal_df
    grouped_df = pd.merge(grouped_df, reported_smiles_signal_df, on='smiles', how='left')

    # pred 1 smiles
    threshold = 0.95
    shared_logger.log(f"using threshold {threshold} to select smiles")
    metadata["threshold"] = threshold
    grouped_df['new_category'] = grouped_df['new_category'].astype(float)
    grouped_df['true_category'] = grouped_df['true_category'].astype(float)
    grouped_df["high_pred_score"] = grouped_df['new_category'].apply(lambda x: 1 if x >= threshold else 0)
    preds_1 = grouped_df[((grouped_df['true_category'].isna()) & (grouped_df['high_pred_score'] == 1)) | (
            grouped_df['true_category'] == 1)]

    ## selected_smiles
    selected_smiles = preds_1["smiles"].to_list()
    shared_logger.log(
        f"{len(selected_smiles)} selected smiles in total, including {', '.join(selected_smiles[:5])}, etc.")
    metadata["selected_smiles"] = selected_smiles

    shared_logger.log("model 1 session ends")


def model_2(metadata):
    shared_logger.log("model 2 session starts")
    ### create selected_smiles
    shared_logger.log("creating selected_smiles")
    selected_smiles = metadata["selected_smiles"]
    selected_smiles_df = pd.DataFrame({"Smiles": selected_smiles})

    ### load common_solvents
    shared_logger.log("loading common_solvents")
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
    prediction_with_progress = PredictionWithProgress()
    wait_thread = threading.Thread(target=prediction_with_progress.print_wait_message, args=("predicting model 2",))
    wait_thread.start()
    model_2_preds_df = predict_model_2(test_path=model_2_data_path,
                                       scaled_preds_path=model_2_scaled_preds_path,
                                       preds_path=model_2_preds_path)
    prediction_with_progress.stop_flag.set()
    wait_thread.join()
    metadata["model_2_preds_df"] = model_2_preds_df

    ## load reported active smiles properties path
    shared_logger.log("loading reported active smiles properties path")
    reported_active_smiles_properties_df = pd.read_csv(reported_active_smiles_properties_path)
    reported_active_smiles_properties_df["properties reported"] = 1
    metadata["reported_active_smiles_properties_df"] = reported_active_smiles_properties_df

    shared_logger.log("model 2 session ends")


def generate_comprehensive_prediction(metadata):
    # Generate Comprehensive Prediction
    shared_logger.log("generating comprehensive prediction")
    model_1_preds_df = metadata["model_1_preds_df"]
    model_1_preds_df.loc[model_1_preds_df['new_category'] == 'Invalid SMILES', 'new_category'] = -1
    model_1_preds_df['new_category'] = model_1_preds_df['new_category'].astype(float)

    grouped_model_1_preds_df = model_1_preds_df.loc[model_1_preds_df.groupby('smiles')['new_category'].idxmax()]
    threshold = metadata["threshold"]
    grouped_model_1_preds_df["high_pred_score"] = grouped_model_1_preds_df['new_category'].apply(
        lambda x: 1 if x >= threshold else 0)

    def write_model_pred(row):
        if row['new_category'] == -1:
            activity_score = "Invalid SMILES"
            model_1_comments = "Invalid SMILES"
        else:
            activity_score = row['new_category']
            model_1_comments = f"absorption_max {row['absorption_max']}, emission_max {row['emission_max']}"
        return pd.Series([activity_score, model_1_comments])

    grouped_model_1_preds_df[['activity_score', 'model_1_comments']] = grouped_model_1_preds_df.apply(write_model_pred,
                                                                                                      axis=1)
    grouped_model_1_preds_df.drop(columns=['absorption_max', 'emission_max', 'new_category'], inplace=True)

    reported_smiles_signal_df = metadata["reported_smiles_signal_df"]
    grouped_model_1_preds_report_df = pd.merge(grouped_model_1_preds_df, reported_smiles_signal_df, on='smiles',
                                               how='left')
    grouped_model_1_preds_report_df['true_category'] = grouped_model_1_preds_report_df['true_category'].fillna("NA")
    grouped_model_1_preds_report_df.rename(columns={"smiles": "smiles_"}, inplace=True)

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
            shared_logger.log(f"generating {output_full_path} ...")
            with pd.ExcelWriter(output_full_path) as writer:
                for i, solvent in enumerate(common_solvents):
                    shared_logger.log(f"\tgenerating sheet {solvent} ...")
                    merged_preds_report_1 = pd.merge(df, grouped_model_1_preds_report_df, left_on=column_name,
                                                     right_on='smiles_', how='left')

                    merged_preds_2 = pd.merge(merged_preds_report_1,
                                              model_2_preds_df[model_2_preds_df['Solvent'] == solvent],
                                              left_on='smiles_', right_on='Smiles', how='left')
                    merged_preds_2["Solvent"] = solvent

                    comprehensive_preds = pd.merge(merged_preds_2, reported_active_smiles_properties_df,
                                                   on=["Smiles", "Solvent"], how='left')
                    comprehensive_preds['properties reported'] = comprehensive_preds['properties reported'].fillna(0)
                    comprehensive_preds = comprehensive_preds.drop(columns=["smiles_", "Smiles", "Solvent"])

                    sheet_name = f"{solvent} ({i + 1})"
                    comprehensive_preds.to_excel(writer, sheet_name=sheet_name, index=False)

                    # workbook = writer.book
                    # worksheet = writer.sheets[sheet_name]
                    # # set the hyperlink
                    # for idx, row in comprehensive_preds.iterrows():
                    #     smiles = row[column_name]
                    #     img_path = os.path.join(smiles_inventory_folder, smiles2path[smiles])
                    #     hyperlink = os.path.join("../../..", smiles_inventory_folder, smiles2path[smiles])
                    #     if os.path.isfile(img_path):
                    #         cell = f'{col_letter}{idx + 2}'
                    #         worksheet[cell].hyperlink = hyperlink
            shutil.copy(output_full_path, os.path.join(app_output_data_folder, f"[method 1] processed {file_name}"))


def generate_report(metadata):
    shared_logger.log("generating report")
    all_input_smiles = set()
    all_model_1_reported_smiles = set()
    all_model_1_reported_positive_smiles = set()
    all_model_1_reported_negative_smiles = set()
    all_model_1_not_reported_smiles = set()
    all_model_1_not_reported_positive_smiles = set()
    all_model_1_not_reported_negative_smiles = set()
    all_model_2_candidates_pairs = set()
    all_model_2_property_reported_pairs = set()
    all_model_2_property_not_reported_pairs = set()
    report = {"file_name": [],
              "input_smiles": [],
              "model_1_reported_smiles": [],
              "model_1_reported_positive_smiles": [],
              "model_1_reported_negative_smiles": [],
              "model_1_not_reported_smiles": [],
              "model_1_not_reported_positive_smiles": [],
              "model_1_not_reported_negative_smiles": [],
              "model_2_candidates_pairs": [],
              "model_2_property_reported_pairs": [],
              "model_2_property_not_reported_pairs": [],
              }
    for file_name in os.listdir(comprehensive_folder):
        shared_logger.log(f"processing {file_name}")

        total_input_smiles = set()
        total_model_1_reported_smiles = set()
        total_model_1_reported_positive_smiles = set()
        total_model_1_reported_negative_smiles = set()
        total_model_1_not_reported_smiles = set()
        total_model_1_not_reported_positive_smiles = set()
        total_model_1_not_reported_negative_smiles = set()
        total_model_2_candidates_pairs = set()
        total_model_2_property_reported_pairs = set()
        total_model_2_property_not_reported_pairs = set()

        if file_name.endswith('.xlsx'):
            full_path = os.path.join(comprehensive_folder, file_name)
            excel_file = pd.ExcelFile(full_path, engine='openpyxl')

            for sheet_name in excel_file.sheet_names:
                shared_logger.log(f"  {sheet_name}")
                solvent = sheet_name.split()[0]
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                df = df[df["activity_score"] != "Invalid SMILES"]
                column_name = [column for column in df.columns if column.lower() == "smiles"][0]
                input_smiles = get_smiles(df, column_name)

                model_1_reported_df = df[~df["true_category"].isna()]
                model_1_reported_positive_df = model_1_reported_df[model_1_reported_df["true_category"] == 1]
                model_1_reported_negative_df = model_1_reported_df[model_1_reported_df["true_category"] == 0]

                model_1_reported_smiles = get_smiles(model_1_reported_df, column_name)
                model_1_reported_positive_smiles = get_smiles(model_1_reported_positive_df, column_name)
                model_1_reported_negative_smiles = get_smiles(model_1_reported_negative_df, column_name)

                model_1_not_reported_df = df[df["true_category"].isna()]
                model_1_not_reported_positive_df = model_1_not_reported_df[
                    model_1_not_reported_df["high_pred_score"] == 1]
                model_1_not_reported_negative_df = model_1_not_reported_df[
                    model_1_not_reported_df["high_pred_score"] == 0]

                model_1_not_reported_smiles = get_smiles(model_1_not_reported_df, column_name)
                model_1_not_reported_positive_smiles = get_smiles(model_1_not_reported_positive_df, column_name)
                model_1_not_reported_negative_smiles = get_smiles(model_1_not_reported_negative_df, column_name)

                model_2_candidates_df = df[
                    (df["true_category"] == 1) | ((df["true_category"].isna()) & (df["high_pred_score"] == 1))]
                model_2_property_reported_df = model_2_candidates_df[model_2_candidates_df["properties reported"] == 1]
                model_2_property_not_reported_df = model_2_candidates_df[
                    model_2_candidates_df["properties reported"] == 0]

                model_2_candidates_pairs = {(smiles, solvent) for smiles in
                                            get_smiles(model_2_candidates_df, column_name)}
                model_2_property_reported_pairs = {(smiles, solvent) for smiles in
                                                   get_smiles(model_2_property_reported_df, column_name)}
                model_2_property_not_reported_pairs = {(smiles, solvent) for smiles in
                                                       get_smiles(model_2_property_not_reported_df, column_name)}

                shared_logger.log(f"\t there are {len(input_smiles)} smiles in input data.")

                shared_logger.log(f"\t\t {len(model_1_reported_smiles)} are reported. "
                                  f"{len(model_1_reported_positive_smiles)} positive, "
                                  f"{len(model_1_reported_negative_smiles)} negative")

                shared_logger.log(f"\t\t {len(model_1_not_reported_smiles)} are not reported. "
                                  f"{len(model_1_not_reported_positive_smiles)} positive, "
                                  f"{len(model_1_not_reported_negative_smiles)} negative")

                shared_logger.log(
                    f"\t there are {len(model_2_candidates_pairs)} (smiles, solvent) pairs are predicted by model 2.")
                shared_logger.log(f"\t\t {len(model_2_property_reported_pairs)} are reported,")
                shared_logger.log(f"\t\t {len(model_2_property_not_reported_pairs)} are not reported.")

                assert len(model_2_candidates_pairs) == len(model_1_reported_positive_smiles) + len(
                    model_1_not_reported_positive_smiles)
                total_input_smiles |= input_smiles
                total_model_1_reported_smiles |= model_1_reported_smiles
                total_model_1_reported_positive_smiles |= model_1_reported_positive_smiles
                total_model_1_reported_negative_smiles |= model_1_reported_negative_smiles
                total_model_1_not_reported_smiles |= model_1_not_reported_smiles
                total_model_1_not_reported_positive_smiles |= model_1_not_reported_positive_smiles
                total_model_1_not_reported_negative_smiles |= model_1_not_reported_negative_smiles
                total_model_2_candidates_pairs |= model_2_candidates_pairs
                total_model_2_property_reported_pairs |= model_2_property_reported_pairs
                total_model_2_property_not_reported_pairs |= model_2_property_not_reported_pairs

        shared_logger.log(f" there are {len(total_input_smiles)} smiles in input data.")

        shared_logger.log(f"\t {len(total_model_1_reported_smiles)} are reported. "
                          f"{len(total_model_1_reported_positive_smiles)} positive, "
                          f"{len(total_model_1_reported_negative_smiles)} negative")

        shared_logger.log(f"\t {len(total_model_1_not_reported_smiles)} are not reported. "
                          f"{len(total_model_1_not_reported_positive_smiles)} positive, "
                          f"{len(total_model_1_not_reported_negative_smiles)} negative")

        shared_logger.log(
            f" there are {len(total_model_2_candidates_pairs)} (smiles, solvent) pairs are predicted by model 2.")
        shared_logger.log(f"\t {len(total_model_2_property_reported_pairs)} are reported,")
        shared_logger.log(f"\t {len(total_model_2_property_not_reported_pairs)} are not reported.")

        report["file_name"].append(file_name)
        report["input_smiles"].append(len(total_input_smiles))
        report["model_1_reported_smiles"].append(len(total_model_1_reported_smiles))
        report["model_1_reported_positive_smiles"].append(len(total_model_1_reported_positive_smiles))
        report["model_1_reported_negative_smiles"].append(len(total_model_1_reported_negative_smiles))
        report["model_1_not_reported_smiles"].append(len(total_model_1_not_reported_smiles))
        report["model_1_not_reported_positive_smiles"].append(len(total_model_1_not_reported_positive_smiles))
        report["model_1_not_reported_negative_smiles"].append(len(total_model_1_not_reported_negative_smiles))
        report["model_2_candidates_pairs"].append(len(total_model_2_candidates_pairs))
        report["model_2_property_reported_pairs"].append(len(total_model_2_property_reported_pairs))
        report["model_2_property_not_reported_pairs"].append(len(total_model_2_property_not_reported_pairs))

        all_input_smiles |= total_input_smiles
        all_model_1_reported_smiles |= total_model_1_reported_smiles
        all_model_1_reported_positive_smiles |= total_model_1_reported_positive_smiles
        all_model_1_reported_negative_smiles |= total_model_1_reported_negative_smiles
        all_model_1_not_reported_smiles |= total_model_1_not_reported_smiles
        all_model_1_not_reported_positive_smiles |= total_model_1_not_reported_positive_smiles
        all_model_1_not_reported_negative_smiles |= total_model_1_not_reported_negative_smiles
        all_model_2_candidates_pairs |= total_model_2_candidates_pairs
        all_model_2_property_reported_pairs |= total_model_2_property_reported_pairs
        all_model_2_property_not_reported_pairs |= total_model_2_property_not_reported_pairs

    shared_logger.log(f"there are {len(all_input_smiles)} smiles in input data.")

    shared_logger.log(f" {len(all_model_1_reported_smiles)} are reported. "
                      f"{len(all_model_1_reported_positive_smiles)} positive, "
                      f"{len(all_model_1_reported_negative_smiles)} negative")

    shared_logger.log(f" {len(all_model_1_not_reported_smiles)} are not reported. "
                      f"{len(all_model_1_not_reported_positive_smiles)} positive, "
                      f"{len(all_model_1_not_reported_negative_smiles)} negative")

    shared_logger.log(
        f"there are {len(all_model_2_candidates_pairs)} (smiles, solvent) pairs are predicted by model 2.")
    shared_logger.log(f" {len(all_model_2_property_reported_pairs)} are reported,")
    shared_logger.log(f" {len(all_model_2_property_not_reported_pairs)} are not reported.")

    report["file_name"].append("all")
    report["input_smiles"].append(len(all_input_smiles))
    report["model_1_reported_smiles"].append(len(all_model_1_reported_smiles))
    report["model_1_reported_positive_smiles"].append(len(all_model_1_reported_positive_smiles))
    report["model_1_reported_negative_smiles"].append(len(all_model_1_reported_negative_smiles))
    report["model_1_not_reported_smiles"].append(len(all_model_1_not_reported_smiles))
    report["model_1_not_reported_positive_smiles"].append(len(all_model_1_not_reported_positive_smiles))
    report["model_1_not_reported_negative_smiles"].append(len(all_model_1_not_reported_negative_smiles))
    report["model_2_candidates_pairs"].append(len(all_model_2_candidates_pairs))
    report["model_2_property_reported_pairs"].append(len(all_model_2_property_reported_pairs))
    report["model_2_property_not_reported_pairs"].append(len(all_model_2_property_not_reported_pairs))

    report_df = pd.DataFrame(data=report)

    report_df.to_csv(report_path, index=False)

    app_output_data_folder = metadata["app_output_data_folder"]
    shutil.copy(report_path, os.path.join(app_output_data_folder, "[method 1] report.csv"))


def method1(metadata):
    load_data(metadata)
    model_1(metadata)
    model_2(metadata)
    generate_comprehensive_prediction(metadata)
    generate_report(metadata)


def predict_model_1(test_path, features_path, preds_path):
    arguments = [
        '--test_path', test_path,
        '--features_path', features_path,
        '--preds_path', preds_path,
        '--checkpoint_dir', model_1_dir,
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)

    t0 = time.time()
    preds = chemprop.train.make_predictions(args=args)
    t1 = time.time()
    shared_logger.log(f"model 1 prediction completed! total time: {t1 - t0} s")
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
    shared_logger.log(f"model 2 prediction completed! total time: {t1 - t0} s")

    df = pd.read_csv(scaled_preds_path)
    df = df[~df.apply(lambda row: row.eq('Invalid SMILES').any(), axis=1)]

    with open(scale_parameters_path, 'r') as json_file:
        scale_parameters = json.load(json_file)

    target_columns = ["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)", "Quantum yield",
                      "log(e/mol-1 dm3 cm-1)", "abs FWHM (cm-1)", "emi FWHM (cm-1)", "abs FWHM (nm)", "emi FWHM (nm)"]
    for target_column in target_columns:
        mean = scale_parameters[target_column]["mean"]
        std_dev = scale_parameters[target_column]["std_dev"]
        df[target_column] = df[f"Scaled {target_column}"].apply(
            lambda x: max(0, x * std_dev + mean) if pd.notna(x) and x != '' else x)
        df = df.drop(f"Scaled {target_column}", axis=1)
    df.to_csv(preds_path, index=False, encoding='utf-8-sig')
    return df
