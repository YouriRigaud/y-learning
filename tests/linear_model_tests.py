"""Test the linear model estimators of ylearn and compare it with sklearn."""

#Author: Youri Rigaud
#License : MIT License

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ylearn.linear_model import OLS
from ylearn.metrics import MSE

def compare_ols() -> bool:
    """
    Compare the OLS classifier model of ylearn with the one from sklearn and compare some metrics.

    Returns:
        bool: True if the metrics show that the ylearn model is as accurate than the sklearn one. 
    """
    print("Test OLS")
    # Load diabetes dataset from sklearn (Regression)
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # ylearn estimator
    ylearn_clf = OLS().fit(X_train, y_train)
    y_pred_ylearn = ylearn_clf.predict(X_test)
    
    # sklearn estimator
    sklearn_clf = LinearRegression().fit(X_train, y_train)
    y_pred_sklearn = sklearn_clf.predict(X_test)

    ylearn_MSE = MSE(y_test, y_pred_ylearn)
    sklearn_MSE = mean_squared_error(y_test, y_pred_sklearn)
    print(f"MSE: ylearn: {ylearn_MSE}; sklearn: {sklearn_MSE}")
    ylearn_r2 = ylearn_clf.score(X_test, y_test)
    sklearn_r2 = sklearn_clf.score(X_test, y_test)
    print(f"R2 score: ylearn: {ylearn_r2}; sklearn: {sklearn_r2}")
    return ylearn_r2 == sklearn_r2 and ylearn_MSE == sklearn_MSE