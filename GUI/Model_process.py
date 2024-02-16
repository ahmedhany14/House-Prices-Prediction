import pandas as pd
import numpy as np
import math
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

from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor


def invese_Scaled_loged_linear_model(Y_prediction):

    for i in range(len(Y_prediction)):
        Y_prediction[i][0] = Y_prediction[i][0] * Y_prediction[i][0]
        Y_prediction[i][0] = math.pow(10, Y_prediction[i][0])
        Y_prediction[i][0] = int(Y_prediction[i][0])

    return Y_prediction


def Scaled_loged_linear_model(X, Y):

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    log_FT = FunctionTransformer(np.log10)
    sqrt_FT = FunctionTransformer(np.sqrt)

    # scaler1 = StandardScaler()

    Y = log_FT.fit_transform(Y)
    Y = sqrt_FT.fit_transform(Y)
    # Y = scaler1.fit_transform(Y)

    return X, Y


def Linear_Scaled_loged_model(
    x_train=pd.DataFrame(),
    y_train=pd.DataFrame(),
):
    # second linear model
    """
    This model will work with scaled and loged data
    which i will first log output data (price)
    then scale them all of data (X, Y)
    """
    CT = ColumnTransformer(
        transformers=[
            ("Log for y_tran", FunctionTransformer(np.log10), []),
            ("SQRT for y_tran", FunctionTransformer(np.sqrt), []),
        ],
        remainder="passthrough",
    )
    pip = Pipeline(
        [
            ("Column Transformer For Y data", CT),
            ("Standard Scaler For x data", StandardScaler()),
            ("Linear Regression ", LinearRegression()),
        ]
    )

    x_train, y_train = Scaled_loged_linear_model(X=x_train, Y=y_train)

    pip.fit(X=x_train, y=y_train)

    return pip


def prediction(X = pd.DataFrame()):

    train = pd.read_csv(
        r"/home/ahmed/Ai/Data science and Ml projects/House-Prices-Prediction---Data-Science-Ml-project/final_cleaned_datasets/Data_set.csv",
    )
    train.drop(columns="Unnamed: 0", inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop("SalePrice", axis=1),
        train["SalePrice"],
        test_size=0.2,
        random_state=100,
    )
    Y = y_train
    y_train = y_train.values.reshape(-1, 1)

    pip = Linear_Scaled_loged_model(x_train, y_train)

    print(X)
    
    X , Y= Scaled_loged_linear_model(X, Y)

    
    predict = pip.predict(X)
    predict = invese_Scaled_loged_linear_model(predict)
    
    return predict[0][0]
    
