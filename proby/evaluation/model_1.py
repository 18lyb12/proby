import chemprop
import os
import pandas as pd
from pathlib import Path

from proby.evaluation.util import classification_evaluation_summary

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
test_full_path = os.path.join(processed_data_folder, 'model_1_test_full.csv')
test_smiles_path = os.path.join(processed_data_folder, 'model_1_test_smiles.csv')
test_features_path = os.path.join(processed_data_folder, 'model_1_test_features.csv')
test_preds_path = os.path.join(processed_data_folder, 'model_1_test_preds.csv')
test_preds_full_path = os.path.join(processed_data_folder, 'model_1_test_preds_full.csv')

save_dir = os.path.join(root_folder_path, 'models/model_1')


def prediction():

    test_df = pd.read_csv(test_full_path)
    test_df[["smiles"]].to_csv(test_smiles_path, index=False, encoding='utf-8-sig')
    test_features_df = test_df[["absorption_max", "emission_max"]]
    test_features_df.to_csv(test_features_path, index=False, encoding='utf-8-sig')

    arguments = [
        '--test_path', test_smiles_path,
        '--preds_path', test_preds_path,
        '--checkpoint_dir', save_dir,
        '--features_path', test_features_path,
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    test_df['pred_category'] = [x[0] for x in preds]
    test_df['true_category'] = test_df['new_category']
    test_df = test_df[test_df['pred_category'] != 'Invalid SMILES']
    test_df['pred_category'] = test_df['pred_category'].astype(float)
    test_df.to_csv(test_preds_full_path, index=False)
    return test_df


def group_by_smiles(df):
    df["absorption_max_category"] = df["absorption_max"].apply(lambda x: int(float(x) / 100))
    df["emission_max_category"] = df["emission_max"].apply(lambda x: int(float(x) / 100))

    def custom_agg(x):
        return 1 if len(set([_ for _ in x if 3 <= _ <= 6])) >= 3 else 0

    grouped_df = df.groupby('smiles').agg({'pred_category': 'max',
                                           'true_category': 'max',
                                           'new_category': 'max',
                                           'absorption_max_category': custom_agg,
                                           'emission_max_category': custom_agg
                                           }).reset_index()
    grouped_df = grouped_df[(grouped_df['new_category'] == 1) | (grouped_df['absorption_max_category'] == 1) | (
                grouped_df['emission_max_category'] == 1)].reset_index()
    return grouped_df


def main():
    prediction()
    df = pd.read_csv(test_preds_full_path)
    classification_evaluation_summary(df['true_category'], df['pred_category'],
                                      roc_fig_path=os.path.join(processed_data_folder, "model_1_ROC.png"),
                                      pr_fig_path=os.path.join(processed_data_folder, "model_1_PR.png"))

    grouped_df = group_by_smiles(df)
    classification_evaluation_summary(grouped_df['true_category'], grouped_df['pred_category'],
                                      roc_fig_path=os.path.join(processed_data_folder, "model_1_grouped_ROC.png"),
                                      pr_fig_path=os.path.join(processed_data_folder, "model_1_grouped_PR.png"))

    df_sorted = df.sort_values(by="pred_category", ascending=False)
    k = 1000
    top = df_sorted.head(k)
    precision = top["true_category"].sum() / len(top)
    print(f"Precision for the top {k} rows: {precision:.2f}")


if __name__ == "__main__":
    main()
