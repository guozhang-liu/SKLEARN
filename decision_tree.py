from sklearn import datasets, preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def decision_tree():
    np.random.RandomState(0)
    win = datasets.load_wine()
    x, y = win.data, win.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    scaler = preprocessing.StandardScaler().fit(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = tree.DecisionTreeClassifier(max_depth=3, criterion="gini")  # 决策树
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print(accuracy_score(y_predict, y_test))
    print(classification_report(y_predict, y_test))
    print(confusion_matrix(y_predict, y_test))


if __name__ == "__main__":
    decision_tree()
