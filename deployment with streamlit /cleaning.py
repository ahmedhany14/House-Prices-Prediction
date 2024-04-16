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
import streamlit as st

pd.options.display.max_columns = None


def remove_column_with_high_missing_values(boston=pd.DataFrame(), combin=[]):
    # remove columns with more than 30% missing values
    uneeded_columns = []

    for column in boston.columns:
        percentage = boston[column].isna().sum() * 100 / len(boston)
        if percentage > 30:
            uneeded_columns.append([column, percentage])
    drop = []
    for i, j in uneeded_columns:
        drop.append(i)

    for dataset in combin:
        dataset.drop(columns=drop, axis=1, inplace=True)

    return combin


# get columns with missing values
def get_null_columns(boston=pd.DataFrame()):
    null_num_columns = []
    null_cat_columns = []

    for column in boston.columns:
        percentage = boston[column].isna().sum() * 100 / len(boston)
        if percentage > 0:
            if boston[column].dtype != "O":
                null_num_columns.append(column)
            else:
                null_cat_columns.append(column)
    return null_num_columns, null_cat_columns


def fill_numerical_values_with_mean(dataset=pd.DataFrame(), column=str):
    mean = dataset[column].mean()
    dataset[column] = dataset[column].fillna(mean)
    return dataset[column]


def fill_numerical_values_with_linear_model(dataset, column):

    data = dataset[[column, "SalePrice"]].copy()
    data[column] = data[column].fillna(-1)
    train = data[data[column] != -1]
    missied_data = pd.DataFrame(data[data[column] == -1]["SalePrice"])

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop(columns=column, axis=1),
        train[column],
        train_size=0.01,
        random_state=42,
    )

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    predction = list(lin_reg.predict(missied_data))

    def update(value):
        if value == -1:
            ret = int(predction[0])
            predction.pop(0)
            return ret
        return value

    dataset[column] = dataset[column].fillna(-1)
    dataset[column] = dataset[column].apply(update)

    return dataset[column]


# filling missing values
def handle_missing_Numerical_values(boston=pd.DataFrame(), num_columns=[]):

    for column in num_columns:
        percentage = boston[column].isna().sum() * 100 / len(boston)
        if percentage <= 3:  # with mean
            boston[column] = fill_numerical_values_with_mean(boston, column)
        else:  # with model
            boston[column] = fill_numerical_values_with_linear_model(boston, column)

    return boston


def fill_numerical_values_with_RF_model(dataset, column):
    data = dataset[[column, "SalePrice"]].copy()
    data[column] = data[column].fillna("missied_data")
    train = data[data[column] != "missied_data"]
    missied_data = pd.DataFrame(data[data[column] == "missied_data"]["SalePrice"])

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop(columns=column, axis=1),
        train[column],
        train_size=0.01,
        random_state=42,
    )

    RF = RandomForestClassifier(ccp_alpha=0.015)
    RF.fit(x_train, y_train)
    predction = list(RF.predict(missied_data))

    def update(value):
        if value == "missied_data":
            ret = predction[0]
            predction.pop(0)
            return ret
        return value

    dataset[column] = dataset[column].fillna("missied_data")
    dataset[column] = dataset[column].apply(update)

    return dataset[column]


def fill_numerical_values_with_mode(dataset, column):
    mode = dataset[column].mode()[0]
    dataset[column] = dataset[column].fillna(mode)
    return dataset[column]


def handle_missing_Categorical_values(boston=pd.DataFrame(), cat_columns=[]):

    for column in cat_columns:
        percentage = boston[column].isna().sum() * 100 / len(boston)
        if percentage <= 3:  # with mean
            boston[column] = fill_numerical_values_with_mode(boston, column)
        else:  # with model
            boston[column] = fill_numerical_values_with_RF_model(boston, column)
    return boston


def Outliers_for_train(boston=pd.DataFrame()):
    numirical_columns = []

    for col in boston.columns:
        if boston[col].dtype != "O":
            numirical_columns.append(col)
    numirical_columns.remove("SalePrice")
    outliers_col = ["LotFrontage", "LotArea", "BsmtFinSF1", "TotalBsmtSF", "GrLivArea"]

    boston = boston.drop(boston[boston["LotFrontage"] > 185].index)
    boston = boston.drop(boston[boston["LotArea"] > 100000].index)
    boston = boston.drop(boston[boston["BsmtFinSF1"] > 4000].index)
    boston = boston.drop(boston[boston["TotalBsmtSF"] > 5000].index)
    boston = boston.drop(boston[boston["GrLivArea"] > 4000].index)

    return boston


def Feature_construction(combin):
    for dataset in combin:
        dataset["Totalarea"] = dataset["LotArea"] + dataset["LotFrontage"]
        dataset["TotalBsmtFin"] = dataset["BsmtFinSF1"] + dataset["BsmtFinSF2"]
        dataset["TotalSF"] = dataset["TotalBsmtSF"] + dataset["2ndFlrSF"]
        dataset["TotalBath"] = dataset["FullBath"] + dataset["HalfBath"]
        dataset["TotalPorch"] = (
            dataset["ScreenPorch"] + dataset["EnclosedPorch"] + dataset["OpenPorchSF"]
        )

    def update(val):
        if val > 0:
            return 1
        return 0

    for dataset in combin:
        dataset["Totalarea"] = dataset["Totalarea"].apply(update)
        dataset["TotalBsmtFin"] = dataset["TotalBsmtFin"].apply(update)
        dataset["TotalSF"] = dataset["TotalSF"].apply(update)
        dataset["TotalBath"] = dataset["TotalBath"].apply(update)
        dataset["TotalPorch"] = dataset["Totalarea"].apply(update)

    return combin


def Feature_selection(boston=pd.DataFrame(), combin=[]):
    removed_columns = set()
    numirical_columns = []
    categorical_columns = []

    for col in boston.columns:
        if boston[col].dtype != "O":
            numirical_columns.append(col)
        else:
            categorical_columns.append(col)

    numirical_columns.remove("SalePrice")

    # VarianceThreshold for numercal columns
    presntage = 0.8 * (1 - 0.8)
    X = boston[numirical_columns]
    var = VarianceThreshold(threshold=presntage)
    var.fit(X)
    boolean_selection = var.get_support()
    columns_names = var.feature_names_in_
    for i in range(len(boolean_selection)):
        if boolean_selection[i] == False:
            removed_columns.add(columns_names[i])

    # strong correlation
    def remove_strong_corr(data_set, neg_corr, pos_corr):
        # global removed_columns
        correlation = data_set.corr()

        for i in range(len(correlation.columns)):

            for j in range(i):
                corr_value = correlation.iloc[i, j]

                if corr_value < 0 and corr_value < neg_corr:
                    print(correlation.columns[j], correlation.columns[i])
                    removed_columns.add(correlation.columns[j])
                elif corr_value > 0 and corr_value > pos_corr:
                    print(correlation.columns[j], correlation.columns[i])
                    removed_columns.add(correlation.columns[j])

        return

    df = boston[numirical_columns].drop(columns=["Id"], axis=1)

    remove_strong_corr(df, neg_corr=-0.4, pos_corr=0.8)
    removed_columns.remove("GrLivArea")

    # Chi2 for categorical columns
    df = boston[categorical_columns]
    ord = OrdinalEncoder()
    ord.fit(df)
    df[df.columns] = ord.transform(df)
    X, Y = df, boston["SalePrice"]
    chi_state, p_value = chi2(X, Y)
    needed_column = 0
    column_names = []
    columns = list(X.columns)

    for i in range(len(p_value)):
        if p_value[i] <= 0.05:
            needed_column += 1
            column_names.append(columns[i])
            removed_columns.add(columns[i])

    # VarianceThreshold for categorical columns
    presntage = 0.8 * (1 - 0.8)
    X = df
    var = VarianceThreshold(threshold=presntage)
    var.fit(X)
    boolean_selection = var.get_support()
    columns_names = var.feature_names_in_
    for i in range(len(boolean_selection)):
        if boolean_selection[i] == False:
            removed_columns.add(columns_names[i])

    # 6. remove columns that have same value more than 95 %
    for col in boston.columns:
        count = boston[col].value_counts().sort_values(ascending=False)
        top_value_count = count.iloc[0]
        if top_value_count * 100 / len(boston) > 85:
            removed_columns.add(col)
    removed_columns.remove("TotalBath")
    removed_columns.remove("TotalSF")
    removed_columns.remove("TotalBsmtSF")
    removed_columns.remove("TotalPorch")
    removed_columns.remove("Totalarea")

    for dataset in combin:
        dataset.drop(columns=list(removed_columns), axis=1, inplace=True)

    return combin
