import chemprop
import numpy as np
import os
import pandas as pd
from pathlib import Path

from proby.evaluation.util import plot_parity

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
test_full_path_template = os.path.join(processed_data_folder, 'model_2_test_full_{}.csv')  # required input file
test_smiles_path_template = os.path.join(processed_data_folder, 'model_2_test_smiles_{}.csv')
test_preds_path_template = os.path.join(processed_data_folder, 'model_2_test_preds_{}.csv')
test_preds_full_path_template = os.path.join(processed_data_folder, 'model_2_test_preds_full_{}.csv')
save_dir_template = os.path.join(root_folder_path, 'models/model_2/{}')

target_list = ['abs', 'emi', 'plqy', 'log10e', 'lifetime', 'abs fwhm (nm)', 'emi fwhm (nm)']

def prediction():
    for target in target_list:
        print(f"========================= start {target} =========================")
        test_full_path = test_full_path_template.format(target)  # required input file
        test_smiles_path = test_smiles_path_template.format(target)
        test_preds_path = test_preds_path_template.format(target)
        test_preds_full_path = test_preds_full_path_template.format(target)

        save_dir = save_dir_template.format(target)

        test_df = pd.read_csv(test_full_path)
        test_df[["smiles", "solvent"]].to_csv(test_smiles_path, index=False, encoding='utf-8-sig')

        arguments = [
            '--test_path', test_smiles_path,
            '--preds_path', test_preds_path,
            '--checkpoint_dir', save_dir,
            '--number_of_molecules', '2',
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args)

        test_df[[f"Pred {target}"]] = preds

        test_df = test_df[~test_df.apply(lambda row: row.eq('Invalid SMILES').any(), axis=1)]
        test_df = test_df[test_df[target].notna()]
        test_df.to_csv(test_preds_full_path, index=False)
    return


def main():
    # generate predictions
    prediction()
    
    # plot parity plots
    for target in target_list:
        test_preds_full_path = os.path.join(processed_data_folder, f'model_2_test_preds_full_{target}.csv')
        df = pd.read_csv(test_preds_full_path)
        sub_df = df[[target, f"Pred {target}"]].dropna()

        if target == "log10e":
            sub_df = sub_df[~(sub_df[target] < 2)]

        fig_path = os.path.join(processed_data_folder, f"model_2 {target} parity plot.png")
        plot_parity(sub_df[target], sub_df[f"Pred {target}"], label=target, fig_path=fig_path)


if __name__ == "__main__":
    main()
