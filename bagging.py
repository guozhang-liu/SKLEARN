from sklearn import datasets, preprocessing, tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings


def bagging():
    warnings.filterwarnings("ignore")
    np.random.RandomState(0)
    win = datasets.load_wine()
    x, y = win.data, win.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    scaler = preprocessing.StandardScaler().fit(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = ensemble.BaggingClassifier(KNeighborsClassifier(), n_estimators=50, max_samples=0.7, max_features=0.7)

    # # bagging表格搜索调参
    # clf = GridSearchCV(ensemble.BaggingClassifier(KNeighborsClassifier()),
    #                    param_grid={"n_estimators": np.arange(1, 51, 5),
    #                                "max_samples": [0.5, 0.6, 0.7, 0.8],
    #                                "max_features": [0.5, 0.6, 0.7, 0.8]
    #                    })

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    # print(clf.best_score_, clf.best_params_)  # 表格搜索结果

    print(accuracy_score(y_predict, y_test))
    print(classification_report(y_predict, y_test))
    print(confusion_matrix(y_predict, y_test))


if __name__ == "__main__":
    bagging()