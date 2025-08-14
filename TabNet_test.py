from A import *
from TabNet import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_id = 0
X_train, X_test, t_train, t_test = LoadData(data_id=data_id,test_size=0.3,noise=0.1)

clf = TabNetClassifierSK(epochs=400, batch_size=None, verbose=1)
clf.fit(X_train, t_train)
preds = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(t_test, preds))