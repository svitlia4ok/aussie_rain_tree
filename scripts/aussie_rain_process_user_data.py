import numpy as np
import pandas as pd

# Preprocesses new data using an existing scaler object
def preprocess_new_data(data: pd.DataFrame, scalerObj):
    # Create a copy of the input DataFrame to avoid modifying the original data
    df = data.copy()
    
    # Convert the 'Date' column to datetime format and extract the month as a new feature
    df['datetime'] = pd.to_datetime(df.Date)
    df['month'] = df.datetime.dt.month
    
    # Drop the original 'Date' and temporary 'datetime' columns
    df.drop(columns=['Date', 'datetime'], inplace=True)
    
    # Get the names of numerical columns and scale them using the provided scaler object
    number_cols = df.columns.to_list()
    df[number_cols] = scalerObj.transform(df)
    
    return df  # Return the preprocessed DataFrame