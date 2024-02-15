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

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor


class inverses:
    '''
    def inverse_scaled_linear_model(Y_prediction, y_actual):
        scaler1 = StandardScaler()
        scaler1.fit(y_actual)
        Y_prediction = scaler1.inverse_transform(Y_prediction)
        return Y_prediction
    '''

    def invese_Scaled_loged_linear_model(Y_prediction):
        
        for i in range (len(Y_prediction)):
            Y_prediction[i][0] = Y_prediction[i][0] * Y_prediction[i][0]
            Y_prediction[i][0] = math.pow(10, Y_prediction[i][0])
            Y_prediction[i][0] = int(Y_prediction[i][0])

        return Y_prediction


class preprocess:

    def scaled_linear_model(X, Y):
        # X & Y (Stander scaling)

        scaler = StandardScaler()
        #scaler1 = StandardScaler()

        X = scaler.fit_transform(X)
        #Y = scaler1.fit_transform(Y)

        return X, Y

    def Scaled_loged_linear_model(X, Y):

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        log_FT = FunctionTransformer(np.log10)
        sqrt_FT = FunctionTransformer(np.sqrt)

        #scaler1 = StandardScaler()

        Y = log_FT.fit_transform(Y)
        Y = sqrt_FT.fit_transform(Y)
        #Y = scaler1.fit_transform(Y)

        return X, Y


class Models:

    def Linear_Scaled_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):
        # first linear model
        """
        this model will work with only scaled data
        which i will scale the x and y data
        """

        pip = Pipeline(
            [
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", LinearRegression()),
            ]
        )

        x_train, y_train = preprocess.scaled_linear_model(X=x_train, Y=y_train)

        pip.fit(X=x_train, y=y_train)

        return pip



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

        x_train, y_train = preprocess.Scaled_loged_linear_model(X=x_train, Y=y_train)

        pip.fit(X=x_train, y=y_train)

        return pip



    def Ridge_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):
        alpha = [0, 0.1, 3.5, 5.5, 8.4, 9, 10.5]

        min_mse, best_alpha = 10, 0
        for i in alpha:
            
            pip = Pipeline(
                [
                    ("Standar dScaler", StandardScaler()),
                    ("Linear Regression ", Ridge(alpha=i)),
                ]
            )
            X, Y = preprocess.scaled_linear_model(x_train, y_train)
            pip.fit(X, Y)
            y_predict = pip.predict(X)

            MAE = mean_absolute_error(y_true=Y, y_pred=y_predict)
            MSE = mean_squared_error(y_true=Y, y_pred=y_predict)
            RMSE = np.sqrt(mean_squared_error(y_true=Y, y_pred=y_predict))
            print("MAE:", MAE)
            print("MSE:", MSE)
            print("RMSE:", RMSE)
            print()

            if MSE < min_mse:
                min_mse = MSE; best_alpha = i

        pip = Pipeline(
            [
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", Ridge(alpha=best_alpha)),
            ]
        )
        pip.fit(X, Y)
        print(best_alpha)

        return pip

    def Lasso_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):

        alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
        min_mse, best_alpha = 10, 0

        for i in alpha:
            
            pip = Pipeline(
                [
                    ("Standar dScaler", StandardScaler()),
                    ("Linear Regression ", Lasso(alpha=i)),
                ]
            )

            X, Y = preprocess.scaled_linear_model(x_train, y_train)
            pip.fit(X, Y)
            y_predict = pip.predict(X)

            
            MAE = mean_absolute_error(y_true=Y, y_pred=y_predict)
            MSE = mean_squared_error(y_true=Y, y_pred=y_predict)
            RMSE = np.sqrt(mean_squared_error(y_true=Y, y_pred=y_predict))
            print("MAE:", MAE)
            print("MSE:", MSE)
            print("RMSE:", RMSE)
            print()

            if MSE < min_mse:
                min_mse = MSE; best_alpha = i

        pip = Pipeline(
            [
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", Lasso(alpha=best_alpha)),
            ]
        )
        pip.fit(X, Y)
        print(best_alpha)

        return pip

    def Ridge_model_log_scaled(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):
        
        alpha = [0, 0.01, 3, 5, 10, 12, 15, 18, 20]
        min_mse, best_alpha = 10, 0

        for i in alpha:


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
                    ("Linear Regression ", Ridge(alpha=i)),
                ]
            )

            X, Y = preprocess.Scaled_loged_linear_model(x_train, y_train)
            pip.fit(X, Y)

            y_predict = pip.predict(X)
            MAE = mean_absolute_error(y_true=Y, y_pred=y_predict)
            MSE = mean_squared_error(y_true=Y, y_pred=y_predict)
            RMSE = np.sqrt(mean_squared_error(y_true=Y, y_pred=y_predict))
            print("MAE:", MAE)
            print("MSE:", MSE)
            print("RMSE:", RMSE)
            print()

            if MSE < min_mse:
                min_mse = MSE; best_alpha = i
            
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
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", Ridge(alpha=best_alpha)),
            ]
        )
        pip.fit(X, Y)
        print(best_alpha)

        return pip

    def Elastic_Net_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):
        alpha = [0, 0.52, 1, 1.5]
        lambda_ = [0, 0.05, 0.02, 0.03, 0.04]
        min_mse, best_alpha, best_lam = 10, 0, 0

        for i in alpha:
            for j in lambda_:
                
                pip = Pipeline(
                    [
                        ("Standar dScaler", StandardScaler()),
                        ("Linear Regression ", ElasticNet(alpha=i, l1_ratio=j)),
                    ]
                )
                X, Y = preprocess.scaled_linear_model(x_train, y_train)
                pip.fit(X, Y)
                y_predict = pip.predict(X)
                
                MAE = mean_absolute_error(y_true=Y, y_pred=y_predict)
                MSE = mean_squared_error(y_true=Y, y_pred=y_predict)
                RMSE = np.sqrt(mean_squared_error(y_true=Y, y_pred=y_predict))

                if MSE < min_mse:
                    min_mse = MSE; best_alpha = i; best_lam = j

        pip = Pipeline(
            [
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", ElasticNet(alpha=best_alpha, l1_ratio=best_lam)),
            ]
        )
        print(best_alpha)
        pip.fit(X, Y)

        return pip

    def DecisionTreeRegressor_Scaled_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):

        pip = Pipeline(
            [
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", DecisionTreeRegressor(random_state=0)),
            ]
        )
        x_train, y_train = preprocess.scaled_linear_model(X=x_train, Y=y_train)

        pip.fit(X=x_train, y=y_train)

        return pip


    def DecisionTreeRegressor_Scaled_loged_model(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):

        pip = Pipeline(
            [
                ("Standar dScaler", StandardScaler()),
                ("Linear Regression ", DecisionTreeRegressor(random_state=0)),
            ]
        )
        x_train, y_train = preprocess.Scaled_loged_linear_model(X=x_train, Y=y_train)

        pip.fit(X=x_train, y=y_train)

        return pip
