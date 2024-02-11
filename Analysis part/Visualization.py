import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, math
import scipy.stats as stats

train_df = pd.read_csv(
    r"/home/ahmed/Ai/Data science and Ml projects/House-Prices-Prediction---Data-Science-Ml-project/original Data sets/train.csv"
)
test_df = pd.read_csv(
    r"/home/ahmed/Ai/Data science and Ml projects/House-Prices-Prediction---Data-Science-Ml-project/original Data sets/test.csv"
)
df = train_df


def price_histplot():
    plt.figure(figsize=(5, 5))

    sns.histplot(data=df, x="SalePrice", stat="density", kde=True, alpha=0.1)
    
    plt.figure(figsize=(5, 5))
    
    stats.probplot(df["SalePrice"], dist="norm", plot=plt)
    
    plt.title("SalePrice QQ Plot")
    plt.show()
    return

def price_histplot_log():
    plt.figure(figsize=(5, 5))

    price = pd.DataFrame()
    price['SalePrice'] = np.log1p(df['SalePrice'])

    sns.histplot(data=price, x="SalePrice", stat="density", kde=True, alpha=0.1)
    plt.figure(figsize=(5, 5))

    stats.probplot(price["SalePrice"], dist="norm", plot=plt)

    plt.title("SalePrice QQ Plot")
    plt.show()
    return


def MSZoning_bar_find_mean_Sales():

    mapper = {
        "RL": "1 Residential Low Density",
        "RM": "2 Residential Medium Density",
        "RH": "3 Residential High Density",
        "FV": "4 Floating Village Residential",
        "C (all)": "5 Commercial",
    }

    def update(kine):
        return mapper[kine]

    MSZoning = pd.DataFrame()
    MSZoning["SalePrice"] = df["SalePrice"]
    MSZoning["MSZoning"] = df["MSZoning"].fillna("RL")
    MSZoning["MSZoning"] = MSZoning["MSZoning"].apply(update)
    order = [
        "1 Residential Low Density",
        "2 Residential Medium Density",
        "3 Residential High Density",
        "4 Floating Village Residential",
        "5 Commercial",
    ]

    plt.figure(figsize=(5, 3))
    sns.barplot(
        data=MSZoning,
        x="MSZoning",
        y="SalePrice",
        order=order,
        estimator="median",
        width=0.2,
        palette="Dark2",
    )
    plt.xticks(rotation=90)
    return


def relation_between_sizes_and_price():

    fig, axe = plt.subplots(1, 4, figsize=(30, 10))

    sns.scatterplot(
        data=df, x="LotFrontage", y="SalePrice", color="r", markers=".", ax=axe[0]
    )
    axe[0].set_label("relation of infront of house size and price")
    axe[0].set_xlabel("infront of house size")
    axe[0].set_ylabel("SalePrice")

    sns.scatterplot(
        data=df, x="LotArea", y="SalePrice", color="blue", markers=".", ax=axe[1]
    )
    axe[1].set_label("relation of house size and price")
    axe[1].set_xlabel("house size")
    axe[1].set_ylabel("SalePrice")

    sns.kdeplot(data=df, x="LotFrontage", y="SalePrice", ax=axe[2])

    sns.kdeplot(data=df, x="LotArea", y="SalePrice", ax=axe[3])

    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    return


def sizes_histplot():

    plt.figure(figsize=(4, 4))

    sns.histplot(data=df, x="LotFrontage", stat="density", kde=True, alpha=0.1)
    
    plt.figure(figsize=(4, 4))
    
    stats.probplot(df["LotFrontage"], dist="norm", plot=plt)
    
    plt.title("LotFrontage QQ Plot")
    
    plt.figure(figsize=(4, 4))

    sns.histplot(data=df, x="LotArea", stat="density", kde=True, alpha=0.1)
    
    plt.figure(figsize=(4, 4))
    
    stats.probplot(df["LotArea"], dist="norm", plot=plt)
    
    plt.title("LotArea QQ Plot")
    
    plt.show()
    return

def sizes_histplot_log():

    sizes = pd.DataFrame()
    
    sizes['LotFrontage'] = np.log(df['LotFrontage'])
    sizes['LotArea'] = np.log(df['LotArea'])


    plt.figure(figsize=(4, 4))

    sns.histplot(data=sizes, x="LotFrontage", stat="density", kde=True, alpha=0.1)
    
    plt.figure(figsize=(4, 4))
    
    stats.probplot(sizes["LotFrontage"], dist="norm", plot=plt)
    
    plt.title("LotFrontage QQ Plot")
    
    plt.figure(figsize=(4, 4))

    sns.histplot(data=sizes, x="LotArea", stat="density", kde=True, alpha=0.1)
    
    plt.figure(figsize=(4, 4))
    
    stats.probplot(sizes["LotArea"], dist="norm", plot=plt)
    
    plt.title("LotArea QQ Plot")
    
    plt.show()
    return



def street():
    fig, axe = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(
        data=df, x="Street", y="SalePrice", width=0.2, palette="Dark2", ax=axe[0]
    )

    sns.scatterplot(data=df, x="Street", y="SalePrice", ax=axe[1])

    plt.legend()
    plt.show()


def land_and_shape():
    fig, axe = plt.subplots(1, 2, figsize=(8, 5))
    
    sns.barplot(
        data=df,
        x='LotShape',
        y='SalePrice',
        ax=axe[0],
        width=.2
    )
    
    sns.barplot(
        data=df,
        x='LandContour',
        y='SalePrice',
        ax=axe[1],
        width=.2
    )
    plt.show()

def quality():
    fig, axe = plt.subplots(1, 2, figsize=(15, 8))

    sns.boxenplot(
        data=df,
        x='OverallQual',
        y='SalePrice',
        ax=axe[0]
    )    
    
    
    sns.boxenplot(
        data=df,
        x='OverallCond',
        y='SalePrice',
        ax=axe[1]
    )    
    
    plt.show()
    
    
def year_built():
    plt.figure(figsize=(20, 8))

    sns.boxenplot(
        data=df,
        x='YearBuilt',
        y='SalePrice',
    )    
    plt.xticks(rotation=90)
    plt.show()
    
    