"""
Lab 4 - Task 4.2 Diabetes Diagnosis using Naive Bayes
Predict diabetes progression CATEGORY from 10 patient features.
Target (continuous) is bucketed into 3 classes: 0-150, 150-300, 300+.
"""
import math
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
Y = [math.floor(v / 150) for v in diabetes.target]   # 0-150 ->0, 150-300 ->1, 300+ ->2

if __name__ == '__main__':
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=1)
    nb = GaussianNB().fit(Xtr, Ytr)
    Yp = nb.predict(Xte)
    print("Classes present:", sorted(set(Y)))
    print("Accuracy:", round(accuracy_score(Yte, Yp), 4))
    print("Confusion matrix:\n", confusion_matrix(Yte, Yp))
