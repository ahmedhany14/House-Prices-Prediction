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
from sklearn.base import BaseEstimator, TransformerMixin

pd.options.display.max_columns = None

# 1 Feature Transforming
"""
1. Removing Redundant Features
    1.1. Remove columns with high missing values
2. Handling massing values
    2.1. Numerical values
    2.2. Categorical values
3. Outliers
"""


class Feature_Transforming(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __remove_high_missing_values(self, X):
        un_wanted_columns, instance = [], len(X)
        for column in X.columns:
            percentage = X[column].isna().sum() * 100 / instance
            if percentage > 30:
                un_wanted_columns.append(column)
        X.drop(columns=un_wanted_columns, axis=1, inplace=True)
        return X

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------
    # for getting numerical columns with null values to handle it
    def __get_numerical_columns(self, X):
        numerical_columns = []
        for col in X.columns:
            null_percentage = X[col].isna().sum() * 100 / len(X)
            if X[col].dtype != "O" and null_percentage > 0:
                numerical_columns.append(col)

        return numerical_columns

    # -------------------------------------------

    # for getting categorical columns with null values to handle it
    def __get_categorical_columns(self, X):
        categorical_columns = []
        for col in X.columns:
            null_percentage = X[col].isna().sum() * 100 / len(X)
            if X[col].dtype == "O" and null_percentage > 0:
                categorical_columns.append(col)

        return categorical_columns

    # -------------------------------------------

    # method for filling numerical columns with mean values
    def __fill_numerical_values_with_mean(self, X, column):
        mean = X[column].mean()
        X[column] = X[column].fillna(mean)
        return X[column]

    # -------------------------------------------

    # method for filling numerical columns with linear model
    # which i will use SalePrice as a train data to predict the missing values
    # the method works as follows:
    """
        1) i will convert the column with missing values to -1, to be able to distinguish between the missing values and the other values
        2) i will train linear model on sale price and the missing column
        3) by using the sale price of the missing column, i will predict the value of missing column
    """

    def __fill_numerical_values_with_linear_model(self, X, column):
        data = X[[column, "SalePrice"]].copy()
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

        X[column] = X[column].fillna(-1)
        X[column] = X[column].apply(update)

        return X[column]

    # this method will use to handel numerical columns with missing values,
    # if the missing values less than 3% it will fill it with the mean value
    # if the missing values more than 3% it will fill it with a linear model, which i will use SalePrice as a train data to predict the missing values
    def __handle_missing_Numerical_values(self, X):
        numerical_columns = self.__get_numerical_columns(X)

        for column in numerical_columns:
            percentage = X[column].isna().sum() * 100 / len(X)
            if percentage <= 3:
                X[column] = self.__fill_numerical_values_with_mean(X, column)
            else:
                X[column] = self.__fill_numerical_values_with_linear_model(X, column)

        return X

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # method for filling numerical columns with random forest model
    # which i will use SalePrice as a train data to predict the missing values
    # the method works as follows:
    """
        1) i will convert the column with missing values to "missied_data", to be able to distinguish between the missing values and the other values
        2) i will train random forest model on sale price and the missing column
        3) by using the sale price of the missing column, i will predict the value of missing column
    """

    def __fill_categorical_values_with_RF_model(self, dataset, column):
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

    # this method will use to handel categorical columns with missing values,
    # if the missing values less than 3% it will fill it with the mode value
    # if the missing values more than 3% it will fill it with a random forest model, which i will use SalePrice as a train data to predict the missing values
    def __handle_missing_Categorical_values(self, X):
        categorical_columns = self.__get_categorical_columns(X)

        for column in categorical_columns:
            percentage = X[column].isna().sum() * 100 / len(X)
            if percentage <= 3:
                X[column] = X[column].fillna(X[column].mode()[0])
            else:
                X[column] = self.__fill_categorical_values_with_RF_model(X, column)

        return X

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __remove_outliers(self, X):
        X = X.drop(X[X["LotFrontage"] > 185].index)
        X = X.drop(X[X["LotArea"] > 100000].index)
        X = X.drop(X[X["BsmtFinSF1"] > 4000].index)
        X = X.drop(X[X["TotalBsmtSF"] > 5000].index)
        X = X.drop(X[X["GrLivArea"] > 4000].index)
        return X

    def transform(self, X, y=None):
        Data_set = X.copy()
        # 1. Removing Redundant Features
        # 1.1. Remove columns with high missing values
        Data_set = self.__remove_high_missing_values(Data_set)

        # 2. Handling massing values
        # 2.1. Numerical values
        Data_set = self.__handle_missing_Numerical_values(Data_set)
        # 2.2. Categorical values
        Data_set = self.__handle_missing_Categorical_values(Data_set)

        # 3. Outliers
        Data_set = self.__remove_outliers(Data_set)
        return Data_set


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2 Feature Construction
class Feature_Construction(BaseEstimator, TransformerMixin):
    def __init__(self):

        pass

    def fit(self, X, y=None):
        return self

    # this method will construct new 5 columns
    """
        1) Totalarea = LotArea + LotFrontage
        2) TotalBsmtFin = BsmtFinSF1 + BsmtFinSF2
        3) TotalSF = TotalBsmtSF + 2ndFlrSF
        4) TotalBath = FullBath + HalfBath
        5) TotalPorch = ScreenPorch + EnclosedPorch + OpenPorchSF
        After constructing them, i will convert them to binary columns, if the value > 0, it will be 1, else it will be 0
    """

    def __feature_construction(self, X):
        X["Totalarea"] = X["LotArea"] + X["LotFrontage"]
        X["TotalBsmtFin"] = X["BsmtFinSF1"] + X["BsmtFinSF2"]
        X["TotalSF"] = X["TotalBsmtSF"] + X["2ndFlrSF"]
        X["TotalBath"] = X["FullBath"] + X["HalfBath"]
        X["TotalPorch"] = X["ScreenPorch"] + X["EnclosedPorch"] + X["OpenPorchSF"]

        def update(val):
            if val > 0:
                return 1
            return 0

        X["Totalarea"] = X["Totalarea"].apply(update)
        X["TotalBsmtFin"] = X["TotalBsmtFin"].apply(update)
        X["TotalSF"] = X["TotalSF"].apply(update)
        X["TotalBath"] = X["TotalBath"].apply(update)
        X["TotalPorch"] = X["Totalarea"].apply(update)
        return X

    def transform(self, X, y=None):
        Data_set = X.copy()
        Data_set = self.__feature_construction(Data_set)
        return Data_set


# 3 Feature Selection
"""
3.1 VarianceThreshold for numercal columns
    for constant columns or columns with low variance (that has more than 80% of the samples)
3.2 strong correlation
3.3 Chi2 for categorical columns
3.4 VarianceThreshold for categorical columns
3.5 remove columns that have same value more than 95 %
3.6 log SalePrice to fix skew 
3.7 Dummy dataset
"""


class Feature_Selection(BaseEstimator, TransformerMixin):
    def __init__(self):
        # these columns i will keep them, because they are important, even if they have low variance or strong correlation, or any other reason
        self.imprortant_columns = [
            "TotalBath",
            "TotalSF",
            "TotalBsmtSF",
            "TotalPorch",
            "Totalarea",
            "GrLivArea",
        ]
        pass

    def fit(self, X, y=None):
        return self

    def __get_numerical_columns(self, X):
        numerical_columns = []
        for col in X.columns:
            if X[col].dtype != "O" and col != "SalePrice":
                numerical_columns.append(col)

        return numerical_columns

    # -------------------------------------------

    # for getting categorical columns with null values to handle it
    def __get_categorical_columns(self, X):
        categorical_columns = []
        for col in X.columns:
            if X[col].dtype == "O":
                categorical_columns.append(col)

        return categorical_columns

    # For removing columns with low variance, based on threshold, which is 0.8 * (1 - 0.8)
    def __VarianceThreshold_for_numercal_columns(self, X):
        numercal_columns = self.__get_numerical_columns(X)
        percetage = 0.8 * (1 - 0.8)
        data_set = X[numercal_columns]
        var_thres = VarianceThreshold(threshold=percetage)
        numercal_columns
        var_thres.fit(data_set)
        boolean_selection = var_thres.get_support()
        columns_names = var_thres.feature_names_in_
        remove = []
        for i in range(len(boolean_selection)):
            if (
                boolean_selection[i] == False
                and columns_names[i] not in self.imprortant_columns
            ):
                remove.append(columns_names[i])
        X.drop(columns=remove, axis=1, inplace=True)
        return X

    # For removing columns with strong correlation
    def __strong_correlation(self, X):
        numercal_columns = self.__get_numerical_columns(X)
        data_set = X[numercal_columns]
        pos_corr, neg_corr, remove = 0.8, -0.4, set()
        correlation = data_set.corr()
        for i in range(len(correlation.columns)):
            for j in range(i):
                corr_value = correlation.iloc[i, j]
                if corr_value < 0 and corr_value < neg_corr:
                    remove.add(correlation.columns[j])
                elif corr_value > 0 and corr_value > pos_corr:
                    remove.add(correlation.columns[j])
        drop = []
        for col in remove:
            if col not in self.imprortant_columns:
                drop.append(col)
        X.drop(columns=drop, axis=1, inplace=True)
        return X

    # For removing columns with Chi2, based on p_value, if p_value <= 0.05, remove the column
    def __Chi2_for_categorical_columns(self, X):
        categorical_columns = self.__get_categorical_columns(X)
        data_set = X[categorical_columns]
        ord = OrdinalEncoder()
        ord.fit(data_set)
        data_set[data_set.columns] = ord.transform(data_set)
        train, target = data_set, X["SalePrice"]
        columns = list(data_set.columns)
        chi_state, p_value = chi2(train, target)
        remove = set()
        for i in range(len(p_value)):
            if p_value[i] <= 0.05 and columns[i] not in self.imprortant_columns:
                remove.add(columns[i])
        X.drop(columns=remove, axis=1, inplace=True)
        return X

    # For removing columns with low variance, based on threshold, which is 0.8 * (1 - 0.8)
    def __VarianceThreshold_for_categorical_columns(self, X):
        categorical_columns = self.__get_categorical_columns(X)
        percetage = 0.8 * (1 - 0.8)
        data_set = X[categorical_columns]
        ord = OrdinalEncoder()
        ord.fit(data_set)
        data_set[data_set.columns] = ord.transform(data_set)
        var_thres = VarianceThreshold(threshold=percetage)
        var_thres.fit(data_set)
        boolean_selection = var_thres.get_support()
        columns_names = var_thres.feature_names_in_
        remove = set()
        for i in range(len(boolean_selection)):
            if (
                boolean_selection[i] == False
                and columns_names[i] not in self.imprortant_columns
            ):
                remove.add(columns_names[i])
        X.drop(columns=remove, axis=1, inplace=True)
        return X

    # For removing columns with same value more than 95%
    def __remove_columns_with_same_value(self, X):
        remove = set()
        for column in X.columns:
            count = X[column].value_counts().sort_values(ascending=False)
            top_value_count = count.iloc[0]
            if (
                top_value_count * 100 / len(X) > 95
                and column not in self.imprortant_columns
            ):
                remove.add(column)
        X.drop(columns=remove, axis=1, inplace=True)
        return X

    # For log SalePrice to fix skew
    def __log_SalePrice(self, X):
        X["SalePrice"] = np.log10(X["SalePrice"])
        return X

    # For Dummy dataset
    def __Dummy_dataset(self, X):
        X = pd.get_dummies(X, drop_first=True)
        for col in X.columns:
            if X[col].dtype == "O":
                X[col] = X[col].astype("int64")
        return X

    def transform(self, X, y=None):
        Data_set = X.copy()

        Data_set = self.__VarianceThreshold_for_numercal_columns(Data_set)
        Data_set = self.__strong_correlation(Data_set)
        Data_set = self.__Chi2_for_categorical_columns(Data_set)
        Data_set = self.__VarianceThreshold_for_categorical_columns(Data_set)
        Data_set = self.__remove_columns_with_same_value(Data_set)
        Data_set = self.__log_SalePrice(Data_set)
        Data_set = self.__Dummy_dataset(Data_set)

        return Data_set

pip = Pipeline(
    [
        ("Feature_Transforming", Feature_Transforming()),
        ("Feature_Construction", Feature_Construction()),
        ("Feature_Selection", Feature_Selection()),
    ]
)
