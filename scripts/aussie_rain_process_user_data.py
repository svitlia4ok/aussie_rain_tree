import numpy as np
import pandas as pd

def preprocess_new_data(data: pd.DataFrame, scalerObj):
    df = data.copy()
    df['datetime'] = pd.to_datetime(df.Date)
    df['month'] = df.datetime.dt.month
    df.drop(columns=['Date', 'datetime'], inplace=True)
    number_cols = df.columns.to_list()
    df[number_cols] = scalerObj.transform(df)
    
    return df
