import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTEN, SMOTENC

def get_inputs_and_targets(df: pd.DataFrame, target_col):
    input_cols = df.drop(target_col, axis=1).columns
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets

def scaleInputs(scalerObj: BaseEstimator, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_scale: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Масштабує числові колонки вхідних даних.

    Args:
        scalerObj (BaseEstimator): Об'єкт скейлера.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.
        number_cols_to_scale (list): Список числових колонок для масштабування.

    Returns:
        dict: Словник з ключами 'inputs_train' і 'inputs_val', що містять масштабовані дані.
    """
    scalerObj.fit(inputs_train[number_cols_to_scale])
    inputs_train[number_cols_to_scale] = scalerObj.transform(inputs_train[number_cols_to_scale])
    inputs_val[number_cols_to_scale] = scalerObj.transform(inputs_val[number_cols_to_scale])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'scalerObj': scalerObj
    }

def imputInputs(imputerObj: BaseEstimator, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_imput: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Масштабує числові колонки вхідних даних.

    Args:
        scalerObj (BaseEstimator): Об'єкт скейлера.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.
        number_cols_to_imput (list): Список числових колонок для імпутації.

    Returns:
        dict: Словник з ключами 'inputs_train' і 'inputs_val', що містять імпутовані дані.
    """
    imputerObj.fit(inputs_train[number_cols_to_imput])
    inputs_train[number_cols_to_imput] = imputerObj.transform(inputs_train[number_cols_to_imput])
    inputs_val[number_cols_to_imput] = imputerObj.transform(inputs_val[number_cols_to_imput])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val
    }

def encodeInputs(encoderObj: BaseEstimator, categorical_cols: List[str], inputs_train: pd.DataFrame, inputs_val: pd.DataFrame) -> Dict[str, Any]:
    """
    Кодує категоріальні колонки вхідних даних.

    Args:
        encoderObj (BaseEstimator): Об'єкт енкодера.
        categorical_cols (list): Список категоріальних колонок для кодування.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.

    Returns:
        dict: Словник з ключами 'inputs_train', 'inputs_val' і 'categories_encoded_cols', що містять закодовані дані та назви закодованих колонок.
    """
    encoderObj.fit(inputs_train[categorical_cols])
    categories_encoded_cols = encoderObj.get_feature_names_out().tolist()
    inputs_train[categories_encoded_cols] = encoderObj.transform(inputs_train[categorical_cols])
    inputs_val[categories_encoded_cols] = encoderObj.transform(inputs_val[categorical_cols])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'categories_encoded_cols': categories_encoded_cols
    }

def processYN(data: pd.DataFrame):
    if 'RainToday' in data.columns:
        data['RainToday'] = data['RainToday'].replace({'Yes': 1, 'No': 0})
    if 'RainTomorrow' in data.columns:
        data['RainTomorrow'] = data['RainTomorrow'].replace({'Yes': 1, 'No': 0})
    return data

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
    
    scalerObj = MinMaxScaler() 
    scaled_data = scaleInputs(scalerObj, inputs_train, inputs_val, number_cols_to_scale)
    inputs_train = scaled_data['inputs_train']
    inputs_val = scaled_data['inputs_val']
    scalerObj = scaled_data['scalerObj']
    
    encoderObj = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = encodeInputs(encoderObj, categorical_cols, inputs_train, inputs_val)
    inputs_train = encoded_data['inputs_train']
    inputs_val = encoded_data['inputs_val']
    categories_encoded_cols = encoded_data['categories_encoded_cols']

    imputerObj = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = imputInputs(imputerObj, inputs_train, inputs_val, number_cols_to_scale)
    inputs_train = imputed_data['inputs_train']
    inputs_val = imputed_data['inputs_val'] 
    
    inputs_train = inputs_train[number_cols + categories_encoded_cols]
    inputs_val = inputs_val[number_cols + categories_encoded_cols]

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
        #'imputerObj': imputerObj,
        'number_cols': number_cols,
        'categorical_cols': categorical_cols,
        'number_cols_to_scale': number_cols_to_scale
    }


def preprocess_new_data(data: pd.DataFrame, scalerObj):
    df = data.copy()
    df['datetime'] = pd.to_datetime(df.Date)
    df['month'] = df.datetime.dt.month
    df.drop(columns=['Date', 'datetime'], inplace=True)
    number_cols = df.columns.to_list()
    df[number_cols] = scalerObj.transform(df)
    
    return df
