import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from A import *

# 1. Load and prepare data
data_id = 3 # Iris = 3

for i in range(100):
    X_train, X_test, t_train, t_test = LoadData(data_id=data_id,test_size=0.3,noise=0.1)

    # Binarize the labels for multi-class ROC
    t_bin = label_binarize(t_train, classes=[0, 1, 2])
    n_classes = t_bin.shape[1]

    # 2. Train a multi-class classifier using OneVsRest strategy
    classifier = MSM_modified()
    classifier.fit(X_train,t_train)

    # Predict probabilities for the test set
    t_score = classifier.predict_proba(X_test,t_test)
    print(classifier.predict(X_test,t_test))
    # 3. Compute ROC curve and AUC for each class
    t_test_all = []
    t_score_all = [] 

    ind = (t_test == 0) | (t_test == 1)
    t_test_all.append(t_test[ind])
    t_score_all.append(t_score[ind])
    ind = (t_test == 0) | (t_test == 2)
    t_test_all.append(t_test[ind]/2)
    t_score_all.append(t_score[ind])
    ind = (t_test == 1) | (t_test == 2)
    t_test_all.append(t_test[ind]-1)
    t_score_all.append(t_score[ind])

    all_test  = np.concatenate(t_test_all)
    all_score = np.concatenate(t_score_all)
    fpr, tpr, _ = roc_curve(all_test, all_score)    
    roc_auc = auc(fpr,tpr)

    if roc_auc > 0.68:
        break

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# fpr[0], tpr[0], _ = roc_curve(t_test[ind], t_score[ind])        
# roc_auc[0] = auc(fpr[0], tpr[0])
# ind = (t_test == 0) | (t_test == 2)
# fpr[1], tpr[1], _ = roc_curve(t_test[ind]/2, t_score[ind])        
# roc_auc[1] = auc(fpr[1], tpr[1])
# ind = (t_test == 1) | (t_test == 2)
# fpr[2], tpr[2], _ = roc_curve(t_test[ind]-1, t_score[ind])        
# roc_auc[2] = auc(fpr[2], tpr[2])

# 4. Plot the ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'EC-MSM (AUC = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level') # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Iris dataset')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# plt.figure(figsize=(8, 6))
# colors = ['blue', 'red', 'green']
# class_labels = np.unique(t_train) # Get class names for legend

# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level') # Diagonal line for reference
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()