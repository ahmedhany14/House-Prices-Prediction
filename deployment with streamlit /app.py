import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import analysis as an
import models as ml
import cleaning as cl
import warnings

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


# ---------------------------------Cleaning and preprocessing---------------------------------#


# This Bolck of code will be contains The Cleaning of the Traing dataset
# in which will contain the following steps:


# ---------------------------------ML Model---------------------------------#


# ---------------------------------User Input---------------------------------#
# This block of code is for the sidebar of the app,
# where the user can select the analysis, cleaning, and model options


# function for getting user input data, which will be taken as a CSV file from user
def user_input_data():

    st.sidebar.header("Upload your CSV data")
    user_input = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if user_input is not None:
        # convring data to csv file
        df = pd.read_csv(user_input)
        # Display the DataFrame
        st.write("Uploaded DataFrame:", df)
        return user_input

    else:
        return None


# loading the data
user_data = user_input_data()


# ---------------------------------User Input prepocessing---------------------------------#


# ---------------------------------User Input Prediction---------------------------------#
