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


def Get_the_numerical_features(boston=pd.DataFrame()):
    # Get the numerical features
    numerical_features = boston.select_dtypes(include=[np.number]).columns
    return numerical_features


# function for displaying the distribution of the columns
def distribution(col, with_log, boston=pd.DataFrame()):
    # Create a figure and axis
    num_rows, num_cols = int(int(len(col) + 1) / 2) + 1, 2
    fig, axe = plt.subplots(num_rows, num_cols, figsize=(10, 10))

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
            dataset[col] = dataset[col] + 1
            dataset[col] = np.log(dataset[col])

        # creating the plot
        sns.histplot(data=dataset, x=feature, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature} by Status")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")

    # i will remove unwanted plots
    for i in range(num_rows):
        for j in range(num_cols):
            if not visited_cells.get((i, j), False):
                axe[i, j].axis("off")

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
