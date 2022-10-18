import pandas as pd
import numpy as np


def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

def drop_colums(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(['ftyrgv','Unnamed: 0', 'nbgde','ID','index'], axis= 1, inplace = True)
    return data

def replace_nan_symbol(data: pd.DataFrame) -> pd.DataFrame:
    data.replace(np.nan, '?', inplace = True)
    return data

def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    return data

def drop_wrong_values(data: pd.DataFrame) -> pd.DataFrame:
    data.replace(['xxxxx', '-26543765432345.0', 'gdhg7u8whui 784gryb', '88876599956788.0'], np.nan, inplace = True)
    data.dropna(inplace=True)
    return data

def replace_symbol_nan(data: pd.DataFrame) -> pd.DataFrame:
    data.replace('?', np.nan, inplace = True)
    data.reset_index(inplace = True, drop = True)
    return data

def change_data_types(data: pd.DataFrame) -> pd.DataFrame:
    columns_names = data.columns.values
    for colum in columns_names:
        data[colum] = data[colum].astype('float64')
    return data

def reset_index(data: pd.DataFrame) -> pd.DataFrame:
    data.reset_index(inplace = True, drop = True)
    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data into useful files for model."""

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_colums)
                    .pipe(replace_nan_symbol)
                    .pipe(drop_duplicates)
                    .pipe(print_shape, msg=' Shape after remove exact duplicates')
                    .pipe(drop_wrong_values)   
                    .pipe(print_shape, msg=' Shape after remove wrong values')   
                    .pipe(replace_symbol_nan) 
                    .pipe(change_data_types)  
                    .pipe(drop_duplicates)
                    .pipe(print_shape, msg=' Shape after remove exact duplicates')
                    )

    return process_data