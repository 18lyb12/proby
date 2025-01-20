import chemprop
import json
import os
import pandas as pd
from pathlib import Path

from proby.evaluation.util import plot_parity

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
common_data_folder = os.path.join(root_folder_path, "data/common")
intermediate_data_folder = os.path.join(root_folder_path, 'data/intermediate')
prediction_full_data_path = os.path.join(processed_data_folder, 'model_2_test_prediction_full_data.csv')

save_dir = os.path.join(root_folder_path, 'models/model_2')
# save_dir =os.path.join(root_folder_path, 'model_candidates/model_2_20250119163759')


def prediction():
    arguments = [
        '--test_path', os.path.join(processed_data_folder, 'model_2_test_data.csv'),
        '--preds_path', os.path.join(processed_data_folder, 'model_2_test_preds.csv'),
        '--checkpoint_dir', save_dir,
        '--number_of_molecules', '2',
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    df = pd.read_csv(os.path.join(processed_data_folder, 'model_2_test_full_data.csv'))
    target_columns = ["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)", "Quantum yield",
                      "log(e/mol-1 dm3 cm-1)", "abs FWHM (cm-1)", "emi FWHM (cm-1)", "abs FWHM (nm)", "emi FWHM (nm)"]
    df[[f"Pred Scaled {target_column}" for target_column in target_columns]] = preds
    df = df[~df.apply(lambda row: row.eq('Invalid SMILES').any(), axis=1)]
    with open(os.path.join(common_data_folder, "scale_parameters.json"), 'r') as json_file:
        scale_parameters = json.load(json_file)
    for target_column in target_columns:
        mean = scale_parameters[target_column]["mean"]
        std_dev = scale_parameters[target_column]["std_dev"]
        df[f"Pred {target_column}"] = df[f"Pred Scaled {target_column}"].apply(
            lambda x: max(0, x * std_dev + mean) if pd.notna(x) and x != '' else x)
    df.to_csv(prediction_full_data_path, index=False)
    return df


def main():
    prediction()
    df = pd.read_csv(prediction_full_data_path)
    for target_column in ["Absorption max (nm)",
                          "Emission max (nm)",
                          "Lifetime (ns)",
                          "Quantum yield",
                          "log(e/mol-1 dm3 cm-1)",
                          "abs FWHM (cm-1)",
                          "emi FWHM (cm-1)",
                          "abs FWHM (nm)",
                          "emi FWHM (nm)"]:
        sub_df = df[[target_column, f"Pred {target_column}"]].dropna()

        fig_path = os.path.join(intermediate_data_folder, f"{target_column.replace('/', '|')} parity plot.png")

        plot_parity(sub_df[target_column], sub_df[f"Pred {target_column}"], label=target_column, fig_path=fig_path)


if __name__ == "__main__":
    main()
