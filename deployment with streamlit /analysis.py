import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings

pd.options.display.max_columns = None
warnings.filterwarnings("ignore")


# functiopn for getting the missing data columns
def Missing_data_columns(boston=pd.DataFrame()):
    null_columns = []

    for col in boston.columns:
        # check if the column has missing values
        if boston[col].isna().sum() > 0:
            null_columns.append([boston[col].isna().sum() * 100 / len(boston), col])
    # sort values by the percentage of missing values descending
    null_columns.sort()
    null_columns.reverse()
    null_columns

    # convert data to DataFrame
    final_result = pd.DataFrame(null_columns, columns=["Percentage", "Column Name"])
    return final_result


# functiopn for getting the numerical features
def Get_the_numerical_features(boston=pd.DataFrame()):
    numerical_features = boston.select_dtypes(include=[np.number]).columns
    return numerical_features


# functiopn for getting the features that has the area in it
def Get_areas_columns(boston=pd.DataFrame()):
    areas = []
    for i in boston.columns:
        if (
            i.find("Area") != -1
            or i.find("Lot") != -1
            or i.find("FinSF") != -1
            or i.find("TotalBsmtSF") != -1
        ):
            areas.append(i)

    return areas


def Get_year_columns(boston=pd.DataFrame()):
    ## get all columns with Year to find the relathon between them and price
    Years = []
    for i in boston.columns:
        if i.find("Year") != -1 or i.find("Yr") != -1:
            Years.append(i)
    return Years


def Get_discrete_numerical_features(boston=pd.DataFrame()):
    numerical_discrete = [
        col
        for col in boston.columns
        if boston[col].dtype != "object"
        and len(boston[col].unique()) < 25
        and col not in Get_year_columns(boston)
    ]
    return numerical_discrete


# function for displaying the distribution of the columns
def distribution(col, with_log, boston=pd.DataFrame()):
    # Create a figure and axis
    num_rows, num_cols = int(int(len(col) + 1) / 2) + 1, 2
    fig, axe = plt.subplots(num_rows, num_cols, figsize=(15, 20))

    # Hash for storing visited cells
    visited_cells = {}
    dataset = boston.copy()

    for i, feature in enumerate(col):

        # Get the row and column index
        row = i // 2
        cols = i % 2
        visited_cells[(row, cols)] = True
        ax = axe[row, cols]

        # if we need to log the data, to avoid the left or right skewness
        if with_log:
            dataset[col] = dataset[col].fillna(0).apply(lambda x: x + 1)
            dataset[col] = np.log(dataset[col])

        # creating the plot
        sns.histplot(data=dataset, x=feature, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")

    # i will remove unwanted plots
    for i in range(num_rows):
        for j in range(num_cols):
            if not visited_cells.get((i, j), False):
                axe[i, j].remove()

    # Display the plot
    plt.tight_layout()
    plt.legend()
    plt.show()
    st.pyplot(fig=fig)
    return


def Heat_map(col, boston=pd.DataFrame()):
    dataset = boston[col].copy()
    corr = dataset.corr()
    fig = plt.figure(figsize=(25, 13))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=plt.cm.CMRmap_r)
    st.pyplot(fig=fig)
    return


def Scater_plot(col, boston=pd.DataFrame()):
    num_rows, num_cols = int(int(len(col) + 1) / 2), 2
    fig, axe = plt.subplots(num_rows, num_cols, figsize=(15, 20))
    dataset = boston.copy()
    for i, feature in enumerate(col):
        row = i // 2
        cols = i % 2
        ax = axe[row, cols]
        sns.scatterplot(data=dataset, x=feature, y="SalePrice", ax=ax)
        ax.set_title(f"{i} with SalePrice")
    plt.tight_layout()
    plt.legend()
    plt.show()
    st.pyplot(fig=fig)
    return


def Lineplot(col, boston=pd.DataFrame()):
    num_rows, num_cols = int(int(len(col) + 1) / 2), 2
    fig, axe = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    dataset = boston.copy()
    for i, col in enumerate(col):
        row = i // 2
        cols = i % 2
        ax = axe[row, cols]
        df = dataset.groupby(col)["SalePrice"].median()
        sns.lineplot(x=df.index, y=df.values, ax=ax)

        plt.xlabel(col)
        ax.set_ylabel("Median House Price")
        ax.set_title("House Price vs YearSold")
    st.pyplot(fig)
    return


def Barplot(col, boston=pd.DataFrame()):
    num_rows, num_cols = int(int(len(col) + 1) / 2), 2
    fig, axe = plt.subplots(num_rows, num_cols, figsize=(10, 20))
    dataset = boston.copy()
    visited_cells = {}

    for i, feature in enumerate(col):
        row = i // 2
        cols = i % 2
        visited_cells[(row, cols)] = True
        ax = axe[row, cols]
        sns.barplot(
            data=dataset,
            x=feature,
            y="SalePrice",
            width=0.5,
            estimator=np.std,
            saturation=0.5,
            errorbar=None,
            ax=ax,
        )
        ax.set_title(f"{feature} vs SalePrice")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel(feature)
        ax.set_ylabel("std of SalePrice")

        # i will remove unwanted plots
    for i in range(num_rows):
        for j in range(num_cols):
            if not visited_cells.get((i, j), False):
                axe[i, j].remove()

    plt.tight_layout()
    plt.legend()
    plt.show()
    st.pyplot(fig=fig)
    return
