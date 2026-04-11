import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
import os
from random import sample
import sys

def calculate_weighted_aupr(label_all, y_score):
    label_all = np.array(label_all).flatten()
    y_score = np.array(y_score).flatten()
    P = np.sum(label_all == 1)
    if P == 0:
        raise ValueError("No positive samples, exit")
    
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = label_all[sorted_indices]
    sorted_scores = y_score[sorted_indices]
    TP = 0  
    FP = 0  
    wp = 0.01
    wn = 1.0
    precision_list = []
    recall_list = []
    
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            TP += 1
        else:
            FP += 1
        
        # Weighted Precision
        weighted_precision = (TP * wp) / (TP * wp + FP * wn) if (TP * wp + FP * wn) > 0 else 0
        
        # Weighted Recall
        recall = TP / P if P > 0 else 0
        
        precision_list.append(weighted_precision)
        recall_list.append(recall)
    
    precision_list = [1.0] + precision_list
    recall_list = [0.0] + recall_list
    
    # Weighted AUPR
    weighted_aupr = 0
    for i in range(1, len(recall_list)):
        recall_diff = recall_list[i] - recall_list[i-1]
        precision_avg = (precision_list[i] + precision_list[i-1]) / 2
        weighted_aupr += recall_diff * precision_avg
    
    return weighted_aupr, precision_list, recall_list

def plot_weighted_pr_curve(precision_list, recall_list, weighted_aupr):

    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list, color='blue', linewidth=2, 
             label=f'Weighted PR Curve (AUPR = {weighted_aupr:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Weighted Precision', fontsize=12)
    plt.title('Weighted Precision-Recall Curve', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def compare_with_standard_pr(label_all, y_score, plot = False):

    weighted_aupr, weighted_precision, weighted_recall = calculate_weighted_aupr(label_all, y_score)
    standard_precision, standard_recall, _ = precision_recall_curve(label_all, y_score)
    standard_aupr = auc(standard_recall, standard_precision)
    print(f"Weighted AUPR: {weighted_aupr:.4f}")
    print(f"Standard AUPR: {standard_aupr:.4f}")
    if not plot:
        return 0
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(weighted_recall, weighted_precision, color='red', linewidth=2, 
             label=f'Weighted PR Curve (AUPR = {weighted_aupr:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Weighted Precision')
    plt.title('Weighted PR Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(standard_recall, standard_precision, color='blue', linewidth=2, 
             label=f'Standard PR Curve (AUPR = {standard_aupr:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Standard PR Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
    return weighted_aupr, standard_aupr

if __name__ == "__main__":
    label_file = 'examples/RF2-PPI_human/pairs/test.tsv'
    score_path = 'results/RF2_test_score'

    df_label = pd.read_csv(label_file, header = None, names = ['p1', 'p2', 'gt_label'], sep = '\t')
    
    df_score_list = []
    for inf in os.listdir(score_path):
        df = pd.read_csv(os.path.join(score_path,inf))
        df_score_list.append(df)

    df_score = pd.concat(df_score_list, ignore_index=True)
    df_all = df_score.groupby(['p1', 'p2'], as_index=False)['score'].mean()

    df_score_final = df_all.merge(df_label, on = ['p1', 'p2'])
    compare_with_standard_pr(df_score_final['gt_label'], df_score_final['score'])