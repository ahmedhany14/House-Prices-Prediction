import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None




def Missing_data_columns(boston = pd.DataFrame()):
    null_columns = []

    for col in boston.columns:
        if boston[col].isna().sum() > 0:
            null_columns.append([boston[col].isna().sum() * 100 / len(boston), col])

    null_columns.sort()
    null_columns.reverse()
    null_columns
    
    final_result = pd.DataFrame(null_columns, columns = ["Percentage", "Column Name"])
    return final_result
        
    
    