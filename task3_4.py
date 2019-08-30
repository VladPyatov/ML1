import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Download data from data-logistic.csv file.
df = pd.read_csv('classification.csv')
y_true = df['true'].values
y_pred = df['pred'].values

# Count TP, FP, FN, and TN according to their definitions.
TP = np.logical_and(y_true==1, y_pred==1).sum()
TN = np.logical_and(y_true==0, y_pred==0).sum()
FP = np.logical_and(y_true==0, y_pred==1).sum()
FN = np.logical_and(y_true==1, y_pred==0).sum()

print(f"1.\n TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")

# Count the basic quality metrics of the classifier.
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"2.\naccuracy score = {acc:.3f}\nprecision score = {pre:.3f}\nrecal score = {rec:.3f}\nf1 score = {f1:.3f}")

# There are four trained classifiers in scores.csv. Load this file.
df = pd.read_csv('scores.csv')

# Calculate the area under the ROC curve for each classifier.
scores = {}
for col in df.columns[1:]:
    scores[col] = roc_auc_score(df['true'], df[col])

print('3.\n', max(scores, key=scores.get))

# Which classifier achieves the greatest accuracy (Precision) with a completeness (Recall) of at least 70%?
m = {}
for col in df.columns[1:]:
    precision, recall, thresholds = precision_recall_curve(df['true'], df[col])
    m[col] = max(precision[recall >= 0.7])

print('4.\n', max(m, key=m.get))
















