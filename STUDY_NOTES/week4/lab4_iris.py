"""
Lab 4 - Task 4.1 Iris Classification using Naive Bayes
Predict iris species from flower features. Gaussian Naive Bayes.
  P(c_i | f1,f2) = alpha * P(f1|c_i) * P(f2|c_i) * P(c_i)   (conditional independence)
Runs with 2 features (as in the lab) AND all 4 features (the report task).
Verified result (random_state=1, 80/20 split): accuracy ~0.97 in both cases.
Run: pip install scikit-learn numpy ; python lab4_iris.py
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()


def run(features, label):
    X = iris.data[:, features]
    Y = iris.target
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=1)
    nb = GaussianNB().fit(Xtr, Ytr)
    Yp = nb.predict(Xte)
    print(f"=== {label} ===")
    print("Accuracy:", round(accuracy_score(Yte, Yp), 4))
    print("Confusion matrix:\n", confusion_matrix(Yte, Yp))
    print("Class priors P(c_i):", np.round(nb.class_prior_, 3))
    print()


if __name__ == '__main__':
    run([0, 2], "2 features (sepal length, petal length)")
    run([0, 1, 2, 3], "ALL 4 features")
