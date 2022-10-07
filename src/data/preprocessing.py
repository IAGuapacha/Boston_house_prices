import pandas as pd

def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)

# def drop_duplicates(data: pd.DataFrame,
#                     drop_cols: list) -> pd.DataFrame:
#     """Drop duplicate rows from data."""
#     data = data.drop_duplicates(subset=drop_cols, keep='first')    
#     return data

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data into useful files for model."""

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after remove exact duplicates')               
                    )

    return process_data