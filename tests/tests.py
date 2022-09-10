import os
import logging

import pandas as pd
import pytest
from omegaconf import OmegaConf
from pathlib import Path
from preprocessing.churn_library import EncoderHelper, DataExploration


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error(f" {err} Testing import_data: The file doesn't appear to have rows and columns")
		raise err

@ pytest.fixture()
def input_data_num_cat():
	data = pd.DataFrame({
		'Marital_Status': ['Married', 'Single', 'Married', 'Unknown', 'Married'],
		'Gender': ['M', 'F', 'M', 'F', 'M'],
		'Customer Age': [45, 51, 67, 28, 72],
		'Credit_Limit': [2607.0, 10195.0, 1867.0, 2733.0, 4716.0],
		'Attrition_Flag': ['Existing Customer', 'Existing Customer', 'Existing Customer',
						   'Attrited Customer', 'Attrited Customer']
	})

	return data


@pytest.fixture()
def config_():
	project_path = Path.cwd()
	config_path = (project_path / 'config' / 'test_config.yaml')
	config = OmegaConf.load(config_path)
	config['DIRECTORIES']['PROJECT_DIR'] = str(project_path)
	return config


def test_eda(input_data_num_cat):
	"""

	:param input_data:
	:return:
	"""
	try:
		explore = DataExploration()
		explore.eda(input_data_num_cat)
	except BaseException as e:
		logging.info(f'{e} error computing eda analysis')

@pytest.fixture
def input_data():
	data = pd.DataFrame({'Marital_Status': ['Married', 'Single', 'Married', 'Unknown', 'Married'],
			'Gender': ['M', 'F', 'M', 'F', 'M'], 'Attrition_Flag': ['Existing Customer', 'Existing Customer',
																	'Existing Customer', 'Attrited Customer',
																	'Attrited Customer']})
	return data


@pytest.fixture
def output_data():
	data = pd.DataFrame({'Marital_Status': ['Married', 'Single', 'Married', 'Unknown', 'Married'],
						 'Gender': ['M', 'F', 'M', 'F', 'M'],  'Churn': [0, 0, 0, 1, 1],
						 'Marital_Status_Churn': [0.33, 0.00, 0.33, 1.00, 0.33],
						 'Gender_Churn': [0.33, 0.50, 0.33, 0.50, 0.33]})
	return data


def test_encoder(input_data, output_data, config_):
	try:
		encoder = EncoderHelper()
		result = encoder.get_encoding(input_data, config_['DATA_INFO'])
		assert result.equals(output_data)
	except BaseException as e:
		logging.error(f'{e} error while testing the EcondHelper')

	try:
		target = config_['DATA_INFO']['NEW_TARGET_COL']
		numeric = ['int64', 'float64']
		assert result.filter(regex=target).dtypes.isin(numeric).all()
	except TypeError as e:
		logging.error(f'{e} variables are not numeric')

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








