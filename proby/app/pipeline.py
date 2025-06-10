import chemprop
import os
import pandas as pd
import shutil
import threading
import time
from functools import reduce
from pathlib import Path

from proby.app.util import get_smiles, load_data, shared_logger, PredictionWithProgress, plot_proby_logo

# Get the absolute path of the directory containing the current file
current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parent

# models
model_1_dir = os.path.join(root_folder_path, '../models/model_1')
model_2_dir_template = os.path.join(root_folder_path, '../models/model_2/{}')

# common
common_data_folder = os.path.join(root_folder_path, "../data/common")
# model 1
absorption_emission_path = os.path.join(common_data_folder, 'absorption_emission.csv')
reported_smiles_signal_path = os.path.join(common_data_folder, 'reported_smiles_signal.csv')
# model 2
common_solvents_path = os.path.join(common_data_folder, 'common_solvents.csv')
reported_active_smiles_properties_path = os.path.join(common_data_folder, 'reported_active_smiles_properties.csv')

# intermediate
intermediate_data_folder = os.path.join(root_folder_path, "../data/prediction_data/intermediate")
os.makedirs(intermediate_data_folder, exist_ok=True)
# model 1
model_1_data_path = os.path.join(intermediate_data_folder, 'model_1_data.csv')
model_1_features_path = os.path.join(intermediate_data_folder, 'model_1_features.csv')
# model 2
model_2_data_path = os.path.join(intermediate_data_folder, 'model_2_data.csv')
model_2_preds_path_template = os.path.join(intermediate_data_folder, 'model_2_preds_{}.csv')

# output
output_data_folder = os.path.join(root_folder_path, "../data/prediction_data/output")
os.makedirs(output_data_folder, exist_ok=True)
# model 1
model_1_preds_path = os.path.join(output_data_folder, 'model_1_preds.csv')
# model 2
model_2_preds_path = os.path.join(output_data_folder, 'model_2_preds.csv')

# comprehensive prediction
comprehensive_folder = os.path.join(output_data_folder, "comprehensive")
os.makedirs(comprehensive_folder, exist_ok=True)

# report
report_folder = os.path.join(output_data_folder, "report")
os.makedirs(report_folder, exist_ok=True)
report_path = os.path.join(report_folder, "report.csv")


def model_1(metadata):
    shared_logger.log("=============================== model 1 session starts ===============================")
    # create smiles
    shared_logger.log("creating smiles")
    smiles_list = metadata["smiles_list"]
    smiles_df = pd.DataFrame({"smiles": smiles_list})

    # load common_absorption_emission_pairs
    shared_logger.log("loading common_absorption_emission_pairs")
    absorption_emission_df = pd.read_csv(absorption_emission_path)

    # crose join
    smiles_df['key'] = 1
    absorption_emission_df['key'] = 1
    model_1_data = pd.merge(smiles_df, absorption_emission_df, on='key', how='outer').drop('key', axis=1)
    model_1_data.to_csv(model_1_data_path, index=False, encoding='utf-8-sig')
    model_1_data[["absorption", "emission"]].to_csv(model_1_features_path, index=False, encoding='utf-8-sig')

    # predict model 1
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

    # load reported smiles signal
    shared_logger.log("loading reported smiles signal")
    reported_smiles_signal_df = pd.read_csv(reported_smiles_signal_path)
    metadata["reported_smiles_signal_df"] = reported_smiles_signal_df
    grouped_df = pd.merge(grouped_df, reported_smiles_signal_df, on='smiles', how='left')

    # group by smiles and add prediction signal
    threshold = 0.95
    shared_logger.log(f"using threshold {threshold} to select smiles")
    metadata["threshold"] = threshold
    grouped_df['new_category'] = grouped_df['new_category'].astype(float)
    grouped_df['true_category'] = grouped_df['true_category'].astype(float)
    grouped_df["high_pred_score"] = grouped_df['new_category'].apply(lambda x: 1 if x >= threshold else 0)
    preds_1 = grouped_df[((grouped_df['true_category'].isna()) & (grouped_df['high_pred_score'] == 1)) | (
            grouped_df['true_category'] == 1)]

    # selected_smiles
    selected_smiles = preds_1["smiles"].to_list()
    shared_logger.log(
        f"{len(selected_smiles)} selected smiles in total, including {', '.join(selected_smiles[:5])}, etc.")
    metadata["selected_smiles"] = selected_smiles

    shared_logger.log("=============================== model 1 session ends ===============================")


def model_2(metadata):
    shared_logger.log("=============================== model 2 session starts ===============================")
    # create selected_smiles
    shared_logger.log("creating selected_smiles")
    selected_smiles = metadata["selected_smiles"]
    selected_smiles_df = pd.DataFrame({"smiles": selected_smiles})

    # load common_solvents
    shared_logger.log("loading common_solvents")
    solvents_df = pd.read_csv(common_solvents_path)
    common_solvents = solvents_df['solvent'].to_list()
    metadata["common_solvents"] = common_solvents

    # cross join
    selected_smiles_df['key'] = 1
    solvents_df['key'] = 1
    model_2_data = pd.merge(selected_smiles_df, solvents_df, on='key', how='outer').drop('key', axis=1)
    model_2_data.to_csv(model_2_data_path, index=False, encoding='utf-8-sig')

    # predict model 2
    prediction_with_progress = PredictionWithProgress()
    wait_thread = threading.Thread(target=prediction_with_progress.print_wait_message, args=("predicting model 2",))
    wait_thread.start()

    model_2_preds_df_list = []
    for target in ['abs', 'emi', 'plqy', 'e', 'log10e', 'lifetime', 'abs fwhm (cm-1)', 'emi fwhm (cm-1)', 'abs fwhm (nm)', 'emi fwhm (nm)']:
        print(f"========================== start to predict {target} ==========================")
        model_2_each_property_preds_df = predict_model_2(test_path=model_2_data_path,
                                                         preds_path_template=model_2_preds_path_template,
                                                         target=target)
        model_2_preds_df_list.append(model_2_each_property_preds_df)

    model_2_preds_df = reduce(lambda left, right: pd.merge(left, right, on=['smiles', 'solvent'], how='outer'),
                              model_2_preds_df_list)

    prediction_with_progress.stop_flag.set()
    wait_thread.join()
    metadata["model_2_preds_df"] = model_2_preds_df

    # load reported active smiles properties path
    shared_logger.log("loading reported active smiles properties path")
    reported_active_smiles_properties_df = pd.read_csv(reported_active_smiles_properties_path)
    reported_active_smiles_properties_df["properties reported"] = 1
    metadata["reported_active_smiles_properties_df"] = reported_active_smiles_properties_df

    shared_logger.log("=============================== model 2 session ends ===============================")


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
            model_1_comments = f"absorption_max {row['absorption']}, emission_max {row['emission']}"
        return pd.Series([activity_score, model_1_comments])

    grouped_model_1_preds_df[['activity_score', 'model_1_comments']] = grouped_model_1_preds_df.apply(write_model_pred,
                                                                                                      axis=1)
    grouped_model_1_preds_df.drop(columns=['absorption', 'emission', 'new_category'], inplace=True)

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
        if file_name.endswith('.xlsx') or file_name.endswith('.csv'):
            full_path = os.path.join(input_data_folder, file_name)
            df = pd.read_excel(full_path, dtype=str) if file_name.endswith('.xlsx') else pd.read_csv(full_path)
            column_name = [column for column in df.columns if column.lower() == "smiles"][0]
            df.rename(columns={column_name: "smiles"}, inplace=True)
            column_name = "smiles"
            # Strip any whitespace from the 'smiles' column
            df[column_name] = df[column_name].apply(lambda x: x.strip() if isinstance(x, str) else x)

            output_full_path = os.path.join(comprehensive_folder, file_name)
            shared_logger.log(f"generating {output_full_path} ...")
            with pd.ExcelWriter(output_full_path) as writer:
                for i, solvent in enumerate(common_solvents):
                    shared_logger.log(f"\tgenerating sheet {solvent} ...")
                    merged_preds_report_1 = pd.merge(df, grouped_model_1_preds_report_df, left_on=column_name,
                                                     right_on='smiles_', how='left', suffixes=('', '_drop'))
                    merged_preds_2 = pd.merge(merged_preds_report_1,
                                              model_2_preds_df[model_2_preds_df['solvent'] == solvent],
                                              left_on='smiles_', right_on='smiles', how='left', suffixes=('', '_drop'))
                    comprehensive_preds = pd.merge(merged_preds_2, reported_active_smiles_properties_df,
                                                   on=["smiles", "solvent"], how='left', suffixes=('', '_drop'))

                    comprehensive_preds['properties reported'] = comprehensive_preds['properties reported'].fillna(0)

                    # Drop extra 'smiles' columns that were added during the merge
                    comprehensive_preds = comprehensive_preds.drop(
                        columns=[col for col in comprehensive_preds.columns if 'smiles_' in col])
                    comprehensive_preds = comprehensive_preds.drop(columns=["solvent"])

                    # Save to Excel with unique sheet names
                    sheet_name = f"{solvent} ({i + 1})"
                    comprehensive_preds.to_excel(writer, sheet_name=sheet_name, index=False)
            shutil.copy(output_full_path, os.path.join(app_output_data_folder, f"processed {file_name}"))


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
    shutil.copy(report_path, os.path.join(app_output_data_folder, "report.csv"))


def process_files(metadata):
    shared_logger.log(f"Starting file processing...")

    load_data(metadata)
    model_1(metadata)
    model_2(metadata)
    generate_comprehensive_prediction(metadata)
    generate_report(metadata)

    shared_logger.log(f"File processing completed.")
    shared_logger.log("*** Refresh the page to download the processed files ***")
    plot_proby_logo()


def predict_model_1(test_path, features_path, preds_path):
    arguments = [
        '--test_path', test_path,
        '--features_path', features_path,
        '--preds_path', preds_path,
        '--checkpoint_dir', model_1_dir,
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)

    t0 = time.time()
    chemprop.train.make_predictions(args=args)
    t1 = time.time()
    shared_logger.log(f"model 1 prediction completed! total time: {t1 - t0} s")
    df = pd.read_csv(preds_path)
    return df


def predict_model_2(test_path, preds_path_template, target):
    shared_logger.log(f"start to predict {target}")
    preds_path = preds_path_template.format(target)
    model_path = model_2_dir_template.format(target)

    arguments = [
        '--test_path', test_path,
        '--preds_path', preds_path,
        '--checkpoint_dir', model_path,
        '--number_of_molecules', '2',
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)

    t0 = time.time()
    chemprop.train.make_predictions(args=args)
    t1 = time.time()
    shared_logger.log(f"model 2 {target} prediction completed! total time: {t1 - t0} s")

    df = pd.read_csv(preds_path)
    df = df[~df.apply(lambda row: row.eq('Invalid SMILES').any(), axis=1)]
    return df
