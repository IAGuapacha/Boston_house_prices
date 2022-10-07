import pandas as pd
import numpy as np
import logging
from pathlib import Path

# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

# funtion to remove columns from data
def drop_cols(data: pd.DataFrame, drop_cols: list = None) -> pd.DataFrame:
    """Drop columns from dataframe"""
    return data.drop(drop_cols, axis=1, inplace = True)

# function to replace np.nan values with '?'
def replace_unknown_values(data: pd.DataFrame, replace_values: np.nan) -> pd.DataFrame:
    """Replace unknown values with '?'"""
    return data.replace(replace_values, '?', inplace = True)

def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)

def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    x = pd.read_csv(f"{input_filepath}/x_train.csv")
    y = pd.read_csv(f"{input_filepath}/y_train.csv")

    data = pd.concat([x, y],axis=1)

    """Process raw data into useful files for model."""
    cols_to_drop = ['ftyrgv',
                      'Unnamed: 0',
                      'nbgde',
                      'ID',
                      'index',
                    ]                      

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_cols, drop_cols = cols_to_drop)
                    .pipe(print_shape, msg=' Shape after drop cols')
                    .pipe(replace_unknown_values, replace_values=np.nan)
                    .pipe(print_shape, msg=' Shape after replace unknown values')
                    )
                    
    x_train = process_data.drop("medv", axis=1)
    y_train = process_data["medv"]

    x_train.to_csv(f'{output_filepath}/x_train_model_input.csv',index=False)
    y_train.to_csv(f'{output_filepath}/y_train_model_input.csv',index=False)
    #End
