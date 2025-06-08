import os
from pathlib import Path

import chemprop
import pandas as pd

from proby.evaluation.util import classification_evaluation_summary

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
test_full_path = os.path.join(processed_data_folder, 'model_1.5_test_full.csv')  # required input file
test_smiles_path = os.path.join(processed_data_folder, 'model_1.5_smiles.csv')
test_preds_path = os.path.join(processed_data_folder, 'model_1.5_test_preds.csv')
test_preds_full_path = os.path.join(processed_data_folder, 'model_1.5_test_preds_full.csv')

save_dir = os.path.join(root_folder_path, 'models/model_1.5')


def prediction():
    test_df = pd.read_csv(test_full_path)
    test_df[["smiles"]].to_csv(test_smiles_path, index=False, encoding='utf-8-sig')

    arguments = [
        '--test_path', test_smiles_path,
        '--preds_path', test_preds_path,
        '--checkpoint_dir', save_dir,
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    test_df['pred_category'] = [x[0] for x in preds]
    test_df['true_category'] = test_df['new_category']
    test_df = test_df[test_df['pred_category'] != 'Invalid SMILES']
    test_df['pred_category'] = test_df['pred_category'].astype(float)
    test_df.to_csv(test_preds_full_path, index=False)
    return test_df


def main():
    prediction()
    df = pd.read_csv(test_preds_full_path)
    classification_evaluation_summary(df['true_category'], df['pred_category'],
                                      roc_fig_path=os.path.join(processed_data_folder, "model_1.5_ROC.png"),
                                      pr_fig_path=os.path.join(processed_data_folder, "model_1.5_PR.png"))

    df_sorted = df.sort_values(by="pred_category", ascending=False)
    k = 1000
    top = df_sorted.head(k)
    precision = top["true_category"].sum() / len(top)
    print(f"Precision for the top {k} rows: {precision:.2f}")


if __name__ == "__main__":
    main()
