
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def plt_roc(test_ytrue, test_ypred, train_ytrue, train_ypred, save_path=None):
    print("Plotting ROC Curve...")
    test_fpr, test_tpr, test_thresh = roc_curve(test_ytrue, test_ypred, drop_intermediate=False)
    test_auc = auc(test_fpr, test_tpr)

    train_fpr, train_tpr, train_thresh = roc_curve(train_ytrue, train_ypred, drop_intermediate=False)
    train_auc = auc(train_fpr, train_tpr)

    plt.subplots(1, 1, figsize=(10, 7))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(test_fpr, test_tpr, label=f'test_auc = {test_auc:.3f}')
    plt.plot(train_fpr, train_tpr, label=f'train_auc = {train_auc:.3f}')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    if save_path is not None:
        print(f'Saving ROC Curve to {save_path}...')
        plt.savefig(save_path)

def plt_prec_recall(ytrue, ypred, save_path=None):
    print("Plotting Precision Recall Curve..." )
    precision, recall, thresh = precision_recall_curve(ytrue, ypred)
    avgprec = average_precision_score(ytrue, ypred)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(precision, recall, label=f'Avg Precision Score = {avgprec:.3f}')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')

    if save_path is not None:
        print(f"Saving Precision Recall Curve to {save_path}")
        plt.savefig(save_path)

def plt_conf_matrix(cm, save_path=None):
    print("Plotting Confusion Matrix...")
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    colors = sns.light_palette('blue', as_cmap=True)
    cm_disp = sns.heatmap(cm, cmap=colors, annot=True, fmt='g', cbar=False)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Real Values')
    
    if save_path is not None:
        print(f'Saving Confusion Matrix to {save_path}')
        plt.savefig(save_path)
