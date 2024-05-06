import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import analysis as an
import models as ml
import cleaning as cl
import warnings
import seaborn as sns
import math
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import cleaning as cl

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
pd.options.display.max_columns = None

pd.options.display.max_columns = None
warnings.filterwarnings("ignore")

# header of the app and description of the app
st.header("House Price App Prediction")
st.write("This App is for predicting house prices in USA")
# uploading the data
train_data = pd.read_csv(
    "/home/ahmed/Ai/Data science and Ml projects/House-Prices-Prediction---Data-Science-Ml-project/Date_set/train.csv"
)

boston_house = train_data.copy()
# ---------------------------------EDA---------------------------------#
# This Bolck of code will be contains The EDA of the Traing dataset
# in which will contain the following steps:
# 1) Display the train_data DataFrame
# 2) Display the shape and columns with missing data of the train_data DataFrame
# 3) Display the distribution of the SalePrice column
# 4) Display the correlation between the Numirecal features and the target
# 5) Displaying a scater plot for the area features with the target, to see the relationship between them, if they are related or not
# 6) Displaying the distribution beteen the area features with the target, to see if they are need to scale or not
# 7) Displaying the relationship between the Year features with the target
# 8) Displaying the relationship between the numerical discrete features with the target

# creating button for displaying the EDA
EDA_button = st.toggle("Display EDA")

if EDA_button:

    # Display the train_data DataFrame
    st.write("### Train Data:", train_data)

    # Display the shape and columns with missing data of the train_data DataFrame
    st.write("##### Train Data Shape:", train_data.shape)
    null_values = an.Missing_data_columns(train_data)
    st.write(
        """ ### Missing Data Columns:""",
        null_values,
    )

    # Visualization Part

    # Display the distribution of the SalePrice column
    sale_price_togle = st.toggle("Sale Price Distribution")
    if sale_price_togle:
        st.write("### Sale Price Distribution")
        an.distribution(["SalePrice"], False, train_data)
        st.write("### Sale Price Distribution with Log")
        an.distribution(["SalePrice"], True, train_data)

    # Display the correlation between the Numirecal features and the target
    correlation_togle = st.toggle(
        "Correlation between Numirecal features and the target"
    )
    if correlation_togle:
        numerica_columns = an.Get_the_numerical_features(train_data)
        st.write("### Correlation between Numirecal features and the target")
        an.Heat_map(numerica_columns, train_data)

    # Displaying a scater plot for the area features with the target, to see the relationship between them, if they are related or not
    area_togle = st.toggle("Scater plot for the area features with the target")
    if area_togle:
        area_columns = an.Get_areas_columns(train_data)
        st.write(
            "### Scater plot for the relationship between area features with the target"
        )
        an.Scater_plot(area_columns, train_data)

    # Displaying the distribution beteen the area features with the target, to see if they are need to scale or not
    Areas_price_togle = st.toggle("Areas Distribution")
    if Areas_price_togle:
        area_columns = an.Get_areas_columns(train_data)
        st.write("### Areas Distribution")
        an.distribution(area_columns, False, train_data)

    # Displaying the relationship between the Year features with the target
    Year_togle = st.toggle("Year relationship with the target")
    if Year_togle:
        year_columns = an.Get_year_columns(train_data)
        st.write("### Year Distribution")
        an.Lineplot(year_columns, train_data)

    # Displaying the relationship between the numerical discrete features with the target
    num_discrete_togle = st.toggle("Numerical Discrete relationship with the target")
    if num_discrete_togle:
        num_discrete_columns = an.Get_discrete_numerical_features(train_data)
        # st.write(num_discrete_columns)
        st.write("### Numerical Discrete relationship with the target")
        an.Barplot(num_discrete_columns, train_data)
# ----------------------------------END EDA----------------------------------#


# function for getting user input data, which will be taken as a CSV file from user
def user_input_data():

    # This block of code is for the sidebar of the app,
    st.sidebar.header("Upload your CSV data")
    user_input = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if user_input is not None:
        # convring data to csv file
        user_input = pd.read_csv(user_input)
        # Display the DataFrame
        if user_input is not None:
            st.write("Uploaded DataFrame:", user_input)
        return user_input

    else:
        return None


# loading the data from user
user_data = user_input_data()
data = {
    "model": [],
    "MAE": [],
    "MSE": [],
    "RMSE": [],
    "train_score": [],
    "test_score": [],
}
models = pd.DataFrame(columns=data)


# ---------------------------------ML Model---------------------------------#
def evaluate(model, model_name, x_train, x_test, y_train, y_test):
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    score_train = r2_score(y_true=y_train, y_pred=train_prediction)

    score_test = r2_score(y_true=y_test, y_pred=test_prediction)
    MAE = mean_absolute_error(y_true=y_test, y_pred=test_prediction)
    MSE = mean_squared_error(y_true=y_test, y_pred=test_prediction)
    RMSE = np.sqrt(mean_absolute_error(y_true=y_test, y_pred=test_prediction))
    data = {
        "model": model_name,
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "train_score": score_train,
        "test_score": score_test,
    }

    return data


def split(data):
    X = data.drop("SalePrice", axis=1)
    Y = data["SalePrice"]
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


process_pip = Pipeline(
    [
        ("feature_transforming", cl.Feature_Transforming()),
        ("feature_construction", cl.Feature_Construction()),
        ("feature_selection", cl.Feature_Selection()),
    ]
)
boston_house = process_pip.fit_transform(boston_house)


class modify_test(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def construct(self, X):
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

    def Dummy(self, X):
        X = pd.get_dummies(X, drop_first=True)
        return X

    def missing(self, X):
        for i in X.columns:
            if X[i].isnull().sum() > 0:
                X[i] = X[i].fillna(X[i].mean())
        return X

    def drop_columns(self, X):
        column = boston_house.columns
        remove = []
        for i in X.columns:
            if i not in column:
                remove.append(i)
        X = X.drop(remove, axis=1)
        return X

    def fill_missing_columns(self, X):
        column = boston_house.columns
        for i in column:
            if i not in X.columns and i != "SalePrice":
                X[i] = 0
        return X

    def rearrange(self, X):
        col = []
        for i in boston_house.columns:
            if i in X.columns:
                col.append(i)
        X = X[col]
        return X

    def transform(self, X):
        X = self.construct(X)
        X = self.Dummy(X)
        X = self.missing(X)
        X = self.drop_columns(X)
        X = self.fill_missing_columns(X)
        X = self.rearrange(X)
        return X


if user_data is not None:
    Out_put = user_data.copy()
    pip = Pipeline([("modify_test", modify_test())])
    user_data = pip.fit_transform(user_data)

    lin_reg = pickle.load(open("linear.pkl", "rb"))
    lin_reg_scaled = pickle.load(open("linear_scaled.pkl", "rb"))
    ridge = pickle.load(open("ridge.pkl", "rb"))
    elastic = pickle.load(open("elastic.pkl", "rb"))
    knn = pickle.load(open("knn.pkl", "rb"))
    svr = pickle.load(open("svr.pkl", "rb"))
    dt = pickle.load(open("dt.pkl", "rb"))
    rf_model = pickle.load(open("random_forest.pkl", "rb"))
    vot_model = pickle.load(open("vot_model.pkl", "rb"))

    x_train, x_test, y_train, y_test = split(boston_house)

    lin_reg.fit(x_train, y_train)
    lin_reg_scaled.fit(x_train, y_train)
    ridge.fit(x_train, y_train)
    elastic.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    svr.fit(x_train, y_train)
    dt.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    vot_model.fit(x_train, y_train)

    models.loc[len(models)] = evaluate(
        lin_reg, "lin reg", x_train, x_test, y_train, y_test
    )
    models.loc[len(models)] = evaluate(
        lin_reg_scaled, "lin reg scaled", x_train, x_test, y_train, y_test
    )
    models.loc[len(models)] = evaluate(ridge, "Ridge", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(
        elastic, "Elastic", x_train, x_test, y_train, y_test
    )
    models.loc[len(models)] = evaluate(knn, "Knn", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(svr, "Svr", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(
        dt, "Decision Tree", x_train, x_test, y_train, y_test
    )
    models.loc[len(models)] = evaluate(
        rf_model, "Random Foreest", x_train, x_test, y_train, y_test
    )
    models.loc[len(models)] = evaluate(
        vot_model, "Votting", x_train, x_test, y_train, y_test
    )

    model_performance = st.toggle("Model Performance")
    if model_performance:
        st.write("### Models Performance on the train data")
        st.write(models.sort_values(by="test_score", ascending=False))
        fig = plt.figure(figsize=(10, 5))
        sns.lineplot(y=models["train_score"], x=models["model"], label="Train Score")
        for i in range(len(models)):
            plt.text(
                i, models["train_score"][i], f'{round(models["train_score"][i],3)}%'
            )

        sns.lineplot(y=models["test_score"], x=models["model"], label="Test Score")

        plt.xticks(rotation=60)

        # displaying the percentage of each model in the plot
        for i in range(len(models)):
            plt.text(i, models["test_score"][i], f'{round(models["test_score"][i],3)}%')
        st.pyplot(fig)
    # ---------------------------------User Input Prediction---------------------------------#

    if user_data is not None:

        def output_inverse(value):
            value_inverse = math.pow(10, value)
            return value_inverse

        predictions = vot_model.predict(user_data)

        for i in range(len(predictions)):
            predictions[i] = output_inverse(predictions[i])

        Out_put["Predictions 📈"] = predictions
        st.write("### Predicted Sale Price for the user data")
        st.write(Out_put)
