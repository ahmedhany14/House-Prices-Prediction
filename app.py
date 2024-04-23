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


# ---------------------------------User Input and Train Preprocessing ---------------------------------#
# Steps:
# 1. removing the columns with high missing values
# 2. Handling massing values for the train data
# 2.1. Numerical values (train_data data set)
# 2.2. Categorical values (train_data data set)
# 2.3. Handling massing values for the user data

# 3.Outliers for train_data
# 4.Feature Construction (construction of new features for both data sets and then convert them to Binay Columns)
# 5.Feature Selection on train data
# 6.log SalePrice to fix skew
# 7.Dummy dataset


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
Output = 0
if user_data is not None:
    Output = user_data.copy()
if user_data is not None:
    # now it's time to clean the data, to train the model on the train data, and predict the out put for the user data
    combin = [train_data, user_data]

    # 1. removing the columns with high missing values
    cl.remove_column_with_high_missing_values(train_data, combin)

    # 2. Handling massing values for the train data

    null_num_columns, null_cat_columns = cl.get_null_columns(train_data)
    # 2.1. Numerical values (train_data data set)
    train_data = cl.handle_missing_Numerical_values(train_data, null_num_columns)
    # 2.2. Categorical values (train_data data set)
    train_data = cl.handle_missing_Categorical_values(train_data, null_cat_columns)

    # 2.3. Handling massing values for the user data
    null_num_columns, null_cat_columns = cl.get_null_columns(user_data)
    for column in null_cat_columns:
        user_data[column] = cl.fill_numerical_values_with_mode(user_data, column)

    for column in null_num_columns:
        user_data[column] = cl.fill_numerical_values_with_mean(user_data, column)

    # 3.Outliers for train_data
    train_data = cl.Outliers_for_train(train_data)
    combin = [train_data, user_data]

    # 4. Feature Construction (construction of new features for both data sets, and then convert them to Binay Columns)
    combin = cl.Feature_construction(combin)

    # 5.Feature Selection on train data
    combin = cl.Feature_selection(train_data, combin)

    # 6. log SalePrice to fix skew
    train_data["SalePrice"] = np.log10(train_data["SalePrice"])

    # 7. Dummy dataset
    train_data = pd.get_dummies(train_data, drop_first=True)
    user_data = pd.get_dummies(user_data, drop_first=True)
    user_data["Exterior1st_ImStucc"] = False
    user_data["Exterior1st_Stone"] = False
    final_user_data = pd.DataFrame()
    for col in train_data.columns:
        if col != "SalePrice":
            final_user_data[col] = user_data[col]
    user_data = final_user_data

    for col in train_data.columns:
        if train_data[col].dtype == "bool":
            train_data[col] = train_data[col].astype("int32")

    for col in user_data.columns:
        if user_data[col].dtype == "bool":
            user_data[col] = user_data[col].astype("int32")

# ---------------------------------ML Model---------------------------------#

if user_data is not None:

    data = {
        "model": [],
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "train_score": [],
        "test_score": [],
    }
    models = pd.DataFrame(columns=data)

    def split(data):
        X = data.drop("SalePrice", axis=1)
        Y = data["SalePrice"]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    x_train, x_test, y_train, y_test = split(train_data)

    def evaluate(model,model_name, x_train, x_test, y_train, y_test):
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

    # Linear Regression model
    lin_reg = ml.Linear_Regression(x_train, y_train)
    # Linear Regression model with scaling
    lin_reg_scaled = ml.Linear_scaled(x_train, y_train)
    # ridge Regression model
    ridge = ml.Ridge_Regression(x_train, y_train)
    # ElasticNet Regression model
    elastic = ml.ElasticNet_Regression(x_train, y_train)
    # Random Forest model
    rf_model = ml.RandomForest(x_train, y_train)
    # Voting model
    vot_model = ml.Voting_system(x_train, y_train)

    models.loc[len(models)] = evaluate(lin_reg,"lin reg", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(lin_reg_scaled,"lin reg scaled", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(ridge,"Ridge", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(elastic,"Elastic", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(rf_model,"Random Foreest", x_train, x_test, y_train, y_test)
    models.loc[len(models)] = evaluate(vot_model,"Votting", x_train, x_test, y_train, y_test)

    model_performance = st.toggle("Model Performance")
    if model_performance:
        st.write("### Models Performance on the train data")
        st.write(models.sort_values(by="test_score", ascending=False))
        fig = plt.figure(figsize=(10, 5))
        sns.lineplot(y=models["test_score"], x=models["model"])
        st.pyplot(fig)
# ---------------------------------User Input Prediction---------------------------------#

    if user_data is not None:
        def output_inverse(value):
            value_inverse = math.pow(10, value)
            return value_inverse


        predictions = vot_model.predict(user_data)

        for i in range(len(predictions)):
            predictions[i] = output_inverse(predictions[i])
        
        Output["Predictions 📈"] = predictions
        st.write("### Predicted Sale Price for the user data")
        st.write(Output)