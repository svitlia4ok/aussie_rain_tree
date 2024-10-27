import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE

# Splits DataFrame into input features and target column
def get_inputs_and_targets(df: pd.DataFrame, target_col):
    input_cols = df.drop(target_col, axis=1).columns
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets

# Scales input data using specified scaler and columns to scale
def scaleInputs(scalerObj, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_scale: List[str]) -> Dict[str, pd.DataFrame]:
    scalerObj.fit(inputs_train[number_cols_to_scale])
    inputs_train[number_cols_to_scale] = scalerObj.transform(inputs_train[number_cols_to_scale])
    inputs_val[number_cols_to_scale] = scalerObj.transform(inputs_val[number_cols_to_scale])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'scalerObj': scalerObj
    }

# Imputes missing values for specified columns in the input data
def imputInputs(imputerObj, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_imput: List[str]) -> Dict[str, pd.DataFrame]:
    imputerObj.fit(inputs_train[number_cols_to_imput])
    inputs_train[number_cols_to_imput] = imputerObj.transform(inputs_train[number_cols_to_imput])
    inputs_val[number_cols_to_imput] = imputerObj.transform(inputs_val[number_cols_to_imput])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val
    }

# Encodes categorical data in input data using the specified encoder
def encodeInputs(encoderObj, categorical_cols: List[str], inputs_train: pd.DataFrame, inputs_val: pd.DataFrame) -> Dict[str, Any]:
    encoderObj.fit(inputs_train[categorical_cols])
    categories_encoded_cols = encoderObj.get_feature_names_out().tolist()
    inputs_train[categories_encoded_cols] = encoderObj.transform(inputs_train[categorical_cols])
    inputs_val[categories_encoded_cols] = encoderObj.transform(inputs_val[categorical_cols])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'categories_encoded_cols': categories_encoded_cols
    }

# Converts "Yes"/"No" values in specific columns to binary (1/0)
def processYN(data: pd.DataFrame):
    if 'RainToday' in data.columns:
        data['RainToday'] = data['RainToday'].replace({'Yes': 1, 'No': 0})
    if 'RainTomorrow' in data.columns:
        data['RainTomorrow'] = data['RainTomorrow'].replace({'Yes': 1, 'No': 0})
    return data

# Performs full data preprocessing, including encoding, scaling, imputing, and optional oversampling
def preprocess_data(data: pd.DataFrame, oversample = False):
    df = data.copy()
    df['datetime'] = pd.to_datetime(df.Date)
    df['month'] = df.datetime.dt.month
    df.drop(columns=['Date', 'datetime'], inplace=True)
    processYN(df)
    inputs, target = get_inputs_and_targets(df, 'RainTomorrow')
    number_cols = inputs.select_dtypes(include="number").columns.to_list()
    
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, target, train_size=0.8, random_state=42, shuffle=True, stratify=target)
    
    number_cols_to_scale = inputs.select_dtypes(include="number").columns.to_list()
    categorical_cols = inputs.select_dtypes(include="object").columns.to_list()
    
    # Scaling numerical columns
    scalerObj = MinMaxScaler() 
    scaled_data = scaleInputs(scalerObj, inputs_train, inputs_val, number_cols_to_scale)
    inputs_train = scaled_data['inputs_train']
    inputs_val = scaled_data['inputs_val']
    scalerObj = scaled_data['scalerObj']
    
    # Encoding categorical columns
    encoderObj = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = encodeInputs(encoderObj, categorical_cols, inputs_train, inputs_val)
    inputs_train = encoded_data['inputs_train']
    inputs_val = encoded_data['inputs_val']
    categories_encoded_cols = encoded_data['categories_encoded_cols']

    # Imputing missing values
    imputerObj = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = imputInputs(imputerObj, inputs_train, inputs_val, number_cols_to_scale)
    inputs_train = imputed_data['inputs_train']
    inputs_val = imputed_data['inputs_val'] 
    
    # Retaining selected columns after encoding
    inputs_train = inputs_train[number_cols + categories_encoded_cols]
    inputs_val = inputs_val[number_cols + categories_encoded_cols]

    # Optional oversampling to balance classes
    if oversample:
        samplerObj = SMOTE(random_state=12)
        inputs_train, targets_train = samplerObj.fit_resample(inputs_train, targets_train)
    
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'targets_train': targets_train,
        'targets_val': targets_val,
        'scalerObj': scalerObj,
        'encoderObj': encoderObj,
        'imputerObj': imputerObj,
        'number_cols': number_cols,
        'categorical_cols': categorical_cols,
        'number_cols_to_scale': number_cols_to_scale
    }