import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
import math


def rename_columns(df=pd.DataFrame):
    mapper = {
        "LotFrontage": "front_house_area",
        "LotArea": "house_area",
        "Street": "street",
        "OverallQual": "quality",
        "YearBuilt": "year_built",
        "YearRemodAdd": "year_remodel_add",
        "GarageYrBlt": "garage_year_build",
        "BsmtQual": "basement_quality",
        "TotalBsmtSF": "basement_area",
        "CentralAir": "central_air_conditioning",
        "GrLivArea": "grade_living_area",
        "WoodDeckSF": "wood_deck_area",
        "Heating": "heating",
    }
    df.rename(columns=mapper, inplace=True)

    return df


def missing_values(df=pd.DataFrame):

    k = int(math.sqrt(len(df)))
    KNN = KNNImputer(missing_values=np.nan, n_neighbors=k, weights="uniform")
    KNN.fit(df[["front_house_area"]])
    df[["front_house_area"]] = KNN.transform(df[["front_house_area"]])

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    simple_imputer.fit(df[["basement_quality"]])
    df[["basement_quality"]] = simple_imputer.transform(df[["basement_quality"]])

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    simple_imputer.fit(df[["garage_year_build"]])
    df[["garage_year_build"]] = simple_imputer.transform(df[["garage_year_build"]])

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    simple_imputer.fit(df[["basement_area"]])
    df[["basement_area"]] = simple_imputer.transform(df[["basement_area"]])

    return df


def remove_outliers_quantile(Q1, Q2, column, df=pd.DataFrame):

    outlier = df[(df[column] < Q1) | (df[column] > Q2)][column]

    column_without_outlier = df[df[column].isin(outlier) == 0]
    print(len(outlier))
    return column_without_outlier
