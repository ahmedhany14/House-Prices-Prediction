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
df = train_df.merge(test_df, how="outer")


# price plotting data
class price_sales:
    def price_histplot():
        sns.histplot(
            data=df,
            x="SalePrice",
            stat="density",
            kde=False,
        )
        plt.show()
        
    def add(): return 10
