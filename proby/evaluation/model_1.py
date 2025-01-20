import chemprop
import os
import pandas as pd
from pathlib import Path

from proby.evaluation.util import classification_evaluation_summary

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
intermediate_data_folder = os.path.join(root_folder_path, 'data/intermediate')
prediction_full_data_path = os.path.join(processed_data_folder, 'model_1_test_prediction_full_data.csv')

save_dir = os.path.join(root_folder_path, 'models/model_1')
# save_dir =os.path.join(root_folder_path, 'model_candidates/model_1_20250119161016')


def prediction():
    arguments = [
        '--test_path', os.path.join(processed_data_folder, 'model_1_test_data.csv'),
        '--preds_path', os.path.join(processed_data_folder, 'model_1_test_preds.csv'),
        '--checkpoint_dir', save_dir,
        '--features_path', os.path.join(processed_data_folder, 'model_1_test_features.csv'),
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    df = pd.read_csv(os.path.join(processed_data_folder, 'model_1_test_full_data.csv'))
    df['pred_category'] = [x[0] for x in preds]
    df['true_category'] = df['new_category']
    df = df[df['pred_category'] != 'Invalid SMILES']
    df['pred_category'] = df['pred_category'].astype(float)
    df.to_csv(prediction_full_data_path, index=False)
    return df


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
                grouped_df['emission_max_category'] == 1)]
    return grouped_df


def main():
    prediction()
    df = pd.read_csv(prediction_full_data_path)
    classification_evaluation_summary(df['true_category'], df['pred_category'],
                                      roc_fig_path=os.path.join(intermediate_data_folder, "model_1_ROC.png"),
                                      pr_fig_path=os.path.join(intermediate_data_folder, "model_1_PR.png"))

    grouped_df = group_by_smiles(df)
    classification_evaluation_summary(grouped_df['true_category'], grouped_df['pred_category'],
                                      roc_fig_path=os.path.join(intermediate_data_folder, "model_1_grouped_ROC.png"),
                                      pr_fig_path=os.path.join(intermediate_data_folder, "model_1_grouped_PR.png"))

    df_sorted = df.sort_values(by="pred_category", ascending=False)
    k = 1000
    top = df_sorted.head(k)
    precision = top["true_category"].sum() / len(top)
    print(f"Precision for the top {k} rows: {precision:.2f}")


if __name__ == "__main__":
    main()
