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


def Columns_transformers(Y=pd.DataFrame()):

    CT = ColumnTransformer(
        transformers=[
            (
                "One Hot Encode",
                OneHotEncoder(
                    categories=[
                        ["Pave", "Grvl"],
                        ['Floor', 'Wall', 'Grav', 'GasW', 'GasA']
                    ],
                    drop="first",
                ),
                ["street", 'heating'],
            ),
            (
                "Ordina Encode",
                OrdinalEncoder(categories=[["N", "Y"], ["Po", "Fa", "TA", "Gd", "Ex"]]),
                ["central_air_conditioning", "basement_quality"],
            ),
        ],
        remainder="passthrough",
    )

    y_transformer = FunctionTransformer(func=np.log)
    
    y_transformer.fit(Y)
    
    return CT, Y


class PipLines_:

    def Linear_Standrized_pipline(
        x_train=pd.DataFrame(),
        y_train=pd.DataFrame(),
    ):

        CT, y_train = Columns_transformers(Y=y_train)
        scaler = StandardScaler()
        lin_reg = DecisionTreeRegressor(random_state = 100)
        pip = Pipeline(
            [
                ("Columns transformer", CT),
                ("Standard Scaler", scaler),
                ("Linear Regression", lin_reg),
            ]
        )
        
        pip.fit(X=x_train, y=y_train)

        return pip


