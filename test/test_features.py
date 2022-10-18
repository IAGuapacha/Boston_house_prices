from ast import Assert
import pytest
import pandas as pd
import numpy as np

#load data
@pytest.fixture
def leer_datos():
    data_test = pd.read_csv('data/interim/x_train.csv')
    return data_test

def test(leer_datos):
    columnas = leer_datos.columns
    assert len(columnas) == 13