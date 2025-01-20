import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_recall_curve, \
    roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score


def classification_evaluation_summary(y_true, y_pred_prob, roc_fig_path="", pr_fig_path=""):
    # Evaluate the model
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    print(f"ROC AUC: {roc_auc:.2f}")

    # Display other metrics like classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    if roc_fig_path:
        plt.savefig(roc_fig_path, dpi=3000)
        print(f"saved ROC curve to {roc_fig_path}")

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    # PR AUC
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.2f}")
    # Plot the precision-recall curve
    plt.figure()
    plt.plot(recall, precision, label=f"Precision-Recall AUC = {pr_auc:.2f}", color="blue")
    plt.title("Precision-Recall (PR) Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    # Set x-axis and y-axis range to (0, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if pr_fig_path:
        plt.savefig(pr_fig_path, dpi=300)
        print(f"saved PR curve to {pr_fig_path}")


def plot_parity(y_true, y_pred, y_pred_unc=None, label="", fig_path=""):
    axmin = min(min(y_true), min(y_pred)) - 0.1 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1 * (max(y_true) - min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    metrics = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR^2 = {r2:.2f}"
    print(f"{label}\n{metrics}")

    plt.figure()
    plt.plot([axmin, axmax], [axmin, axmax], '--k')

    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', alpha=1, elinewidth=1)

    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))

    ax = plt.gca()
    ax.set_aspect('equal')

    at = AnchoredText(metrics, prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.xlabel(f'True {label}')
    plt.ylabel(f'Predicted {label}')
    if fig_path:
        plt.savefig(fig_path, dpi=3000)
        print(f"saved parity plot to {fig_path}")
