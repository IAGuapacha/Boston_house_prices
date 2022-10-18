import pandas as pd
import logging
from joblib import load
from sklearn.metrics import r2_score
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import numpy as np

# libraries to import function from other folder
import sys
import os
sys.path.append(os.path.abspath('src/'))


from features.build_features import (print_shape, drop_outliers, reset_index, fill_nan,drop_nan)


def main(input_filepath, output_filepath, input_test_filepath, report_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating ML model')

    model = load(f'{output_filepath}/SVR_final_model.joblib')
    
    x_train = pd.read_csv(f"{input_filepath}/x_train_model_input.csv")
    y_train = pd.read_csv(f"{input_filepath}/y_train_model_input.csv")

    y_pred = model.predict(x_train)

    train_score = r2_score(y_train, y_pred)
    print(f"Train Score: {train_score}")

    with open(f'{report_filepath}/train_score.txt', 'w') as f:
        f.write(f"Train reacall Score: {train_score}")

    # test predictions

    x_test = pd.read_csv(f"{input_test_filepath}/x_test.csv")
    y_test = pd.read_csv(f"{input_test_filepath}/y_test.csv")

    test = pd.concat([x_test, y_test],axis = 1)

    test_eval = feature_process(test)

    x_test_model = test_eval.drop("medv", axis=1)
    y_test_model = test_eval["medv"]

    y_test_pred = model.predict(x_test_model)

    test_score = r2_score(y_test_model, y_test_pred)
    print(f"Test Score: {test_score}")

    with open(f'{report_filepath}/test_score.txt', 'w') as f:
        f.write(f"Test recall Score: {test_score}")    



def feature_process(data: pd.DataFrame) -> pd.DataFrame:

                            
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
                    .pipe(reset_index)
                    )
    return process_data                


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(f'{project_dir}/data/processed', 
        f'{project_dir}/models',
        f'{project_dir}/data/interim', 
        f'{project_dir}/reports')