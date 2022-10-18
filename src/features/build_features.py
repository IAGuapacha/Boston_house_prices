import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path

def getQuartiles(dataFrame, column):
    Q1 = np.percentile(dataFrame[column].dropna(), 25, interpolation = 'midpoint')
    Q3 = np.percentile(dataFrame[column].dropna(), 75, interpolation = 'midpoint')

    IQR = Q3 - Q1

    return {'Q1': Q1,'Q3': Q3, 'IQR': IQR}

def getBounds(dataFrame, column, quartiles):
    Q1 = quartiles['Q1']
    Q3 = quartiles['Q3']
    IQR = quartiles['IQR']

    #print('Q1', Q1)
    #print('Q3', Q3)
    #print('IQR', IQR)

    # Upper bound
    upper = np.where(dataFrame[column] >= (Q3 + 1.5*IQR))

    # Below Lower bound
    lower = np.where(dataFrame[column] <= (Q1 - 1.5*IQR))

    return {'upper': upper, 'lower': lower}

def deleteOutliersDataframe(dataFrame, bounds):
    upper = bounds['upper']
    lower = bounds['lower']

    dataFrame.drop(upper[0], inplace = True)
    dataFrame.drop(lower[0], inplace = True)
    dataFrame.reset_index(inplace = True, drop = True)

    return dataFrame

def deleteOutliersSetTrain(train, column):
    quartiles = getQuartiles(train.copy(), column)
    bounds = getBounds(train.copy(), column, quartiles)
    train = deleteOutliersDataframe(train.copy(), bounds)

    return train

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

def drop_outliers(data: pd.DataFrame, col: str) -> pd.DataFrame:
    data = deleteOutliersSetTrain(data,col)
    return data

def fill_nan(data: pd.DataFrame, col:str,value: float ) -> pd.DataFrame:
    data[col].replace(np.nan,value,inplace=True)
    return data

def drop_nan(data: pd.DataFrame, col:str) -> pd.DataFrame:
    data.drop(data[data[col].notnull() == False].index, inplace=True)
    data.reset_index(inplace = True, drop = True)
    return data

def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    return data

def reset_index(data: pd.DataFrame) -> pd.DataFrame:
    data.reset_index(inplace = True, drop = True)
    return data

def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    x = pd.read_csv(f"{input_filepath}/x_train.csv")
    y = pd.read_csv(f"{input_filepath}/y_train.csv")
    print(x.info())
    print(y.info())

    data = pd.concat([x, y],axis=1)
                    

    process_data = (data
                    .pipe(drop_outliers, col ="crim")  
                    .pipe(print_shape, msg=' Shape after remove outliers from crim')     
                    .pipe(fill_nan, col= "crim",value= 1.27038)
                    .pipe(fill_nan, col= "zn",value= 0.0) 
                    .pipe(drop_nan, col= "indus") 
                    .pipe(print_shape, msg=' Shape after removing nan values from indus') 
                    .pipe(fill_nan, col= "chas", value= 0.0) 
                    .pipe(drop_outliers, col ="nox") 
                    .pipe(print_shape, msg=' Shape after remove outliers from nox')   
                    .pipe(drop_nan, col= "nox") 
                    .pipe(print_shape, msg=' Shape after removing nan values from nox') 
                    .pipe(drop_outliers, col ="rm") 
                    .pipe(print_shape, msg=' Shape after remove outliers from rm') 
                    .pipe(drop_nan, col= "rm") 
                    .pipe(print_shape, msg=' Shape after removing nan values from rm') 
                    .pipe(drop_nan, col= "age") 
                    .pipe(print_shape, msg=' Shape after removing nan values from age') 
                    .pipe(fill_nan, col= "dis", value= 4.2820) 
                    .pipe(drop_outliers, col ="dis") 
                    .pipe(print_shape, msg=' Shape after remove outliers from dis') 
                    .pipe(fill_nan, col= "rad", value= 4.0)
                    .pipe(fill_nan, col= "tax", value= 666.0)
                    .pipe(fill_nan, col= "ptratio", value= 20.2)
                    .pipe(fill_nan, col= "black", value= 396.9)
                    .pipe(drop_outliers, col ="black")
                    .pipe(print_shape, msg=' Shape after remove outliers from black') 
                    .pipe(fill_nan, col= "lstat", value= 1.73)
                    .pipe(drop_outliers, col ="lstat")
                    .pipe(print_shape, msg=' Shape after remove outliers from lstat') 
                    .pipe(drop_nan, col= "lstat") 
                    .pipe(print_shape, msg=' Shape after removing nan values from lstat') 
                    .pipe(drop_nan, col= "medv") 
                    .pipe(drop_duplicates)
                    .pipe(reset_index)
                    )
                    
    x_train = process_data.drop("medv", axis=1)
    y_train = process_data["medv"]

    x_train.to_csv(f'{output_filepath}/x_train_model_input.csv',index=False)
    y_train.to_csv(f'{output_filepath}/y_train_model_input.csv',index=False)
    #End


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(f'{project_dir}/data/interim', f'{project_dir}/data/processed')