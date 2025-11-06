"""Test the KNN estimator of ylearn and compare it with sklearn."""

#Author: Youri Rigaud
#License : MIT License

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from ylearn.neighbors import KNNClassifier
from ylearn.neighbors import KNNRegressor
from ylearn.metrics import MSE

def compare_knn_classifier() -> bool:
    """
    Compare the KNN classifier model of ylearn with the one from sklearn and compare some metrics.

    Returns:
        bool: True if the metrics show that the ylearn model is as accurate than the sklearn one. 
    """
    print("Test KNN classifier")
    # Load breast cancer dataset from sklearn (Classification)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # ylearn estimator
    ylearn_clf = KNNClassifier(k=5).fit(X_train, y_train)
    y_pred_ylearn = ylearn_clf.predict(X_test)

    # sklearn estimator
    sklearn_clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_pred_sklearn = sklearn_clf.predict(X_test)

    ylearn_accuracy = ylearn_clf.score(X_test, y_test)
    sklearn_accuracy = sklearn_clf.score(X_test, y_test)
    print(f"Accuracy: ylearn: {ylearn_accuracy}; sklearn: {sklearn_accuracy}")
    return ylearn_accuracy == sklearn_accuracy

def compare_knn_regressor() -> bool:
    """
    Compare the KNN regressor model of ylearn with the one from sklearn and compare some metrics.

    Returns:
        bool: True if the metrics show that the ylearn model is as accurate than the sklearn one. 
    """
    print("Test KNN regressor")
    # Load diabetes dataset from sklearn (Regressor)
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # ylearn estimator
    ylearn_clf = KNNRegressor(k=5).fit(X_train, y_train)
    y_pred_ylearn = ylearn_clf.predict(X_test)

    # sklearn estimator
    sklearn_clf = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    y_pred_sklearn = sklearn_clf.predict(X_test)

    ylearn_MSE = MSE(y_test, y_pred_ylearn)
    sklearn_MSE = mean_squared_error(y_test, y_pred_sklearn)
    print(f"MSE: ylearn: {ylearn_MSE}; sklearn: {sklearn_MSE}")
    ylearn_r2 = ylearn_clf.score(X_test, y_test)
    sklearn_r2 = sklearn_clf.score(X_test, y_test)
    print(f"R2 score: ylearn: {ylearn_r2}; sklearn: {sklearn_r2}")
    return ylearn_r2 == sklearn_r2 and ylearn_MSE == sklearn_MSE
