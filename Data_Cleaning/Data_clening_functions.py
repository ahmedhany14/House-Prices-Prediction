import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
import math


def rename_columns(df=pd.DataFrame):
    mapper = {
        "LotArea": "house_area",
        "OverallQual": "quality",
        "YearBuilt": "year_built",
        "YearRemodAdd": "year_remodel_add",
        "GarageYrBlt": "garage_year_build",
        "TotalBsmtSF": "basement_area",
        "GrLivArea": "grade_living_area",
        "FullBath": "number_of_bathrooms",
        "Fireplaces": "has_Fireplaces_or_not",
        "GarageArea": "garage_area",
        "GarageCars": "garage_capacite",
        "TotRmsAbvGrd": "Total_rooms",
    }
    df = df.rename(columns=mapper)

    return df


def missing_values(df=pd.DataFrame):
    columns = []

    for i in df.columns:
        if df[i].isna().sum() > 0:
            columns.append(i)

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    simple_imputer.fit(df[columns])
    df[columns] = simple_imputer.transform(df[columns])

    return df


def remove_outliers_quantile(Q1, Q2, column, df=pd.DataFrame):

    outlier = df[(df[column] < Q1) | (df[column] > Q2)][column]

    column_without_outlier = df[df[column].isin(outlier) == 0]
    print(len(outlier))
    return column_without_outlier
