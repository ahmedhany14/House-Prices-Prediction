import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor


def preprocesse_scaled_linear_model(X, Y):
    # X & Y (Stander scaling)

    scaler = StandardScaler()
    scaler1 = StandardScaler()

    X = scaler.fit_transform(X)
    Y = scaler1.fit_transform(Y)

    return X, Y


class Models:

    def Linear_Scaled_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):
        # first linear model
        '''
            this model will work with only scaled data
            which i will scale the x and y data
        '''
        
        x_train, y_train = preprocesse_scaled_linear_model(X=x_train, Y=y_train)
        
        linear_regression = LinearRegression()
        linear_regression.fit(X=x_train,y=y_train)

        return linear_regression
        
        
