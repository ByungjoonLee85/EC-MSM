from A import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load and prepare data
data_id = 3 # Three Crescents = 1

all_preds = []
all_tests = []

n_trials = 100

for _ in range(n_trials):
    # Load data for this trial
    X_train, X_test, t_train, t_test = LoadData(
        n_samples=80, 
        data_id=data_id,
        test_size=0.3, 
        noise=0.1
    )
    
    # Fit the classifier
    classifier = MSM_modified()
    classifier.fit(X_train, t_train)
    
    # Predict for this trial
    t_pred = classifier.predicted_value(X_test, t_test)
    
    # Append results
    all_preds.append(t_pred)
    all_tests.append(t_test)

# Flatten into 1-D arrays
t_pred_all = np.concatenate(all_preds)
t_test_all = np.concatenate(all_tests)

cm = confusion_matrix(t_test_all,t_pred_all,normalize='true')

# Display the confusion matrix
# Using ConfusionMatrixDisplay is the recommended way in scikit-learn
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['setosa', 'versicolor', 'virginica'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Normalized confusion matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()

# You can also print the raw confusion matrix
print("Confusion Matrix:")
print(cm)

