import chemprop
import os
import pandas as pd
from pathlib import Path

from proby.evaluation.util import classification_evaluation_summary

current_file_path = Path(__file__).resolve()
root_folder_path = current_file_path.parents[1]
processed_data_folder = os.path.join(root_folder_path, 'data/processed_data')
intermediate_data_folder = os.path.join(root_folder_path, 'data/intermediate')
prediction_full_data_path = os.path.join(processed_data_folder, 'model_1.5_test_prediction_full_data.csv')

save_dir = os.path.join(root_folder_path, 'models/model_1.5')
# save_dir =os.path.join(root_folder_path, 'model_candidates/model_1.5_20250119162130')


def prediction():
    arguments = [
        '--test_path', os.path.join(processed_data_folder, 'model_1.5_test_data.csv'),
        '--preds_path', os.path.join(processed_data_folder, 'model_1.5_test_preds.csv'),
        '--checkpoint_dir', save_dir,
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    df = pd.read_csv(os.path.join(processed_data_folder, 'model_1.5_test_full_data.csv'))
    df['pred_category'] = [x[0] for x in preds]
    df['true_category'] = df['new_category']
    df = df[df['pred_category'] != 'Invalid SMILES']
    df['pred_category'] = df['pred_category'].astype(float)
    df.to_csv(prediction_full_data_path, index=False)
    return df


def main():
    prediction()
    df = pd.read_csv(prediction_full_data_path)
    classification_evaluation_summary(df['true_category'], df['pred_category'],
                                      roc_fig_path=os.path.join(intermediate_data_folder, "model_1.5_ROC.png"),
                                      pr_fig_path=os.path.join(intermediate_data_folder, "model_1.5_PR.png"))

    df_sorted = df.sort_values(by="pred_category", ascending=False)
    k = 1000
    top = df_sorted.head(k)
    precision = top["true_category"].sum() / len(top)
    print(f"Precision for the top {k} rows: {precision:.2f}")


if __name__ == "__main__":
    main()
