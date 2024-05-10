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
from sklearn.neighbors import KNeighborsRegressor

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

import cleaning as cl

pd.options.display.max_columns = None

feature_transforming = cl.Feature_Transforming()
feature_selection = cl.Feature_Selection()
feature_construction = cl.Feature_Construction()


def Linear_Regression():
    """
    This function is for the Linear Regression model
    """
    lin_reg = LinearRegression()
    pip = Pipeline(
        [
            ("LinearRegression", lin_reg),
        ]
    )
    return pip


def Linear_scaled():
    """
    This function is for the Linear Regression model with scaling
    """
    pip = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("LinearRegression", LinearRegression(positive=True)),
        ]
    )
    return pip


def Ridge_Regression():
    """
    This function is for the Ridge Regression model
    """

    ridge = Ridge(alpha=0.5)

    pip = Pipeline(
        [
            ("Ridge", ridge),
        ]
    )
    return pip


def ElasticNet_Regression():
    """
    This function is for the ElasticNet Regression model
    """
    elastic = ElasticNet(max_iter=5000, alpha=0.0005, l1_ratio=0.5)
    pip = Pipeline(
        [
            ("ElasticNet", elastic),
        ]
    )
    return pip


def DecisionTree():
    """
    This function is for the Decision Tree model
    """
    dt = DecisionTreeRegressor(
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        ccp_alpha=1e-05,
        random_state=45,
    )

    pip = Pipeline(
        [
            ("DecisionTree", dt),
        ]
    )

    return pip


def Knn():
    """
    This function is for the Knn model
    """
    knn = KNeighborsRegressor(n_neighbors=5)

    pip = Pipeline(
        [
            ("Knn", knn),
        ]
    )

    return knn


def RandomForest():
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

    pip = Pipeline(
        [
            ("RandomForest", rf),
        ]
    )

    return rf


def Svr():
    """
    This function is for the Support Vector Regression model
    """
    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

    pip = Pipeline(
        [
            ("SVR", svr),
        ]
    )

    return pip


def Voting_system():
    linear = Linear_Regression()
    linear_scaled = Linear_scaled()
    ridge = Ridge_Regression()
    elastic = ElasticNet_Regression()
    dt = DecisionTree()
    random_forest = RandomForest()
    knn = Knn()
    svr = Svr()

    vot = VotingRegressor(
        estimators=[
            ("LinearRegression", linear),
            ("LinearRegression scaled ", linear_scaled),
            ("Ridge", ridge),
            ("ElasticNet", elastic),
            ("RandomForestRegressor", random_forest),
        ]
    )
    return vot


linear = Linear_Regression()
linear_scaled = Linear_scaled()
ridge = Ridge_Regression()
elastic = ElasticNet_Regression()
random_forest = RandomForest()
dt = DecisionTree()
knn = Knn()
svr = Svr()
vot_model = Voting_system()

import pickle

with open("linear.pkl", "wb") as f:
    pickle.dump(linear, f)

with open("linear_scaled.pkl", "wb") as f:
    pickle.dump(linear_scaled, f)

with open("ridge.pkl", "wb") as f:
    pickle.dump(ridge, f)

with open("elastic.pkl", "wb") as f:
    pickle.dump(elastic, f)

with open("knn.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("svr.pkl", "wb") as f:
    pickle.dump(svr, f)

with open("dt.pkl", "wb") as f:
    pickle.dump(dt, f)

with open("random_forest.pkl", "wb") as f:
    pickle.dump(random_forest, f)

with open("vot_model.pkl", "wb") as f:
    pickle.dump(vot_model, f)
