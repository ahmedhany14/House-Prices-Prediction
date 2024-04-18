import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)

pd.options.display.max_columns = None


def Linear_Regression(x_train, y_train):
    """
    This function is for the Linear Regression model
    """
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    return lin_reg


def Linear_scaled(x_train, y_train):
    pip = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("LinearRegression", LinearRegression(positive=True)),
        ]
    )
    pip.fit(x_train, y_train)
    return pip


def Ridge_Regression(x_train, y_train):
    """
    This function is for the Ridge Regression model
    """
    ridge = Ridge(alpha=0.5)
    ridge.fit(x_train, y_train)
    return ridge


def ElasticNet_Regression(x_train, y_train):
    """
    This function is for the ElasticNet Regression model
    """
    elastic = ElasticNet(max_iter=5000, alpha=0.0005, l1_ratio=0.5)
    elastic.fit(x_train, y_train)
    return elastic


def RandomForest(x_train, y_train):
    """
    This function is for the Random Forest model
    """
    rf = RandomForestRegressor(
        n_estimators=50,
        random_state=45,
        ccp_alpha=1e-05,
        max_depth=30,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=5,
    )
    rf.fit(x_train, y_train)
    return rf


def Voting_system(x_train, y_train):
    vot = VotingRegressor(
        estimators=[
            ("LinearRegression", LinearRegression()),
            (
                "LinearRegression scaled ",
                Pipeline(
                    [
                        ("Scaling", StandardScaler()),
                        ("LinearRegression", LinearRegression(positive=True)),
                    ]
                ),
            ),
            ("Ridge", Ridge(alpha=5)),
            ("ElasticNet", ElasticNet(max_iter=5000, alpha=0.0005, l1_ratio=0.5)),
            (
                "RandomForestRegressor",
                RandomForestRegressor(
                    n_estimators=50,
                    random_state=45,
                    ccp_alpha=1e-05,
                    max_depth=30,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    min_samples_split=5,
                ),
            ),
        ]
    )
    vot.fit(x_train, y_train)
    return vot
