"""
Testing Functions
"""
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from omegaconf import OmegaConf
from churn_library import DataExploration, EncoderHelper
from utils_ import split_data, train_model, import_data
from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import LogisticRegression

# Configure logging
with open(Path(Path.cwd().parent, 'config', 'logging.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

logging.config.dictConfig(config)


def test_import():
    try:
        data = import_data(Path(Path.cwd().parent, "data/bank_data.csv"))
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing import_eda: The file wasn't found {}".format(err))
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(" {} Testing import_data: The file doesn't "
                      " appear to have rows and columns".format(err))


@pytest.fixture()
def input_data_num_cat():
    data = pd.DataFrame({
        'Marital_Status': ['Married', 'Single', 'Married',
                           'Unknown', 'Married'],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Customer Age': [45, 51, 67, 28, 72],
        'Credit_Limit': [2607.0, 10195.0, 1867.0, 2733.0, 4716.0],
        'Attrition_Flag': ['Existing Customer', 'Existing Customer',
                           'Existing Customer', 'Attrited Customer',
                           'Attrited Customer']
    })

    return data


@pytest.fixture()
def config_test():
    project_path = Path.cwd().parent
    config_path = (project_path / 'config' / 'test_config.yaml')
    config = OmegaConf.load(config_path)
    config['DIRECTORIES']['PROJECT_DIR'] = str(project_path)
    return config


def test_eda(input_data_num_cat):
    try:
        explore = DataExploration()
        explore.eda(input_data_num_cat)
    except BaseException as err:
        logging.info('{} error computing eda analysis {}'.format(err))


@pytest.fixture
def input_data():
    data = pd.DataFrame({'Marital_Status': ['Married', 'Single', 'Married',
                                            'Unknown', 'Married'],
                         'Gender': ['M', 'F', 'M', 'F', 'M'],
                         'Attrition_Flag': ['Existing Customer',
                                            'Existing Customer',
                                            'Existing Customer',
                                            'Attrited Customer',
                                            'Attrited Customer']})
    return data


@pytest.fixture
def output_data():
    data = pd.DataFrame({'Marital_Status': ['Married', 'Single', 'Married',
                                            'Unknown', 'Married'],
                         'Gender': ['M', 'F', 'M', 'F', 'M'],
                         'Churn': [0, 0, 0, 1, 1],
                         'Marital_Status_Churn': [0.33, 0.00, 0.33, 1.00,
                                                  0.33],
                         'Gender_Churn': [0.33, 0.50, 0.33, 0.50, 0.33]})
    return data


def test_encoder(input_data, output_data, config_test):
    try:
        encoder = EncoderHelper()
        result = encoder.get_encoding(input_data, config_test['DATA_INFO'])
        assert result.equals(output_data)
    except BaseException as err:
        logging.error('{} error while testing the EcondHelper'.format(err))

    try:
        target = config_test['DATA_INFO']['NEW_TARGET_COL']
        numeric = ['int64', 'float64']
        assert pd.DataFrame(result).filter(regex=target).dtypes.isin(numeric).all()
    except TypeError as err:
        logging.error('{} variables are not numeric'.format(err))


@pytest.fixture
def input_data3():
    data = {
        'Customer_Age': [45, 49, 51, 40, 40, 44, 51, 32, 37, 48],
        'Gender_Churn': [0.33, 0.50, 0.33, 0.50, 0.33, 0.50, 0.33,
                         0.33, 0.50, 0.50],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'F'],
        'Churn': [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)


def test_split(input_data3, config_test):
    try:
        x_train, x_test, y_train, y_test = split_data(
            input_data3,
            config_test['DATA_INFO']['NEW_TARGET_COL'],
            config_test['TEST_SIZE'], config_test['RANDOM_STATE'])
        assert x_train.shape == (7, 3)
        assert x_test.shape == (3, 3)
        assert (type(y_train), type(y_test)) == (pd.Series, pd.Series)
        assert (len(y_train), len(y_test)) == (7, 3)
        logging.info('Testing split succcesful')
    except KeyError as err:
        logging.error('Unable to test {}'.format(err))


@pytest.fixture
def data_sample(config_test):
    df = import_data(Path(Path.cwd().parent, "data/bank_data.csv"))
    return split_data(df, config_test['DATA_INFO']['NEW_TARGET_COL'],
                      config_test['TEST_SIZE'], config_test['RANDOM_STATE'])

def test_train_models(data_sample):

    try:
        x_train, x_test, y_train, y_test = data_sample
        lrc = LogisticRegression(max_iter=100)
        fitted_model, predictions = train_model(x_train, y_test, y_train, lrc,
                                                save_dir=None)
        assert_array_almost_equal(np.round(fitted_model.intercept_, 3),
                                  [0.035], decimal=1)
        assert_array_almost_equal(fitted_model.coef_, [
            [-7.52866129e-03, 3.05745489e-01, 1.91205707e-02,
             -3.69219615e-01, 5.59305942e-01, 6.92126822e-01,
             -3.00673620e-04, -5.96828369e-04, 2.96154754e-04,
             -1.71901823e-01, 3.03169297e-04, -8.64173480e-02,
             -2.21152249e-01, -7.52902453e-03, 9.73271897e-03,
             8.18308479e-03, 9.75816419e-03, 7.38483958e-03,
             6.83536439e-03]], decimal=1)
        assert len(predictions) == 300
        assert max(predictions), min(predictions) == (0, 1)

    except BaseException as err:
        logging.error('model fitting {}'.format(err))

