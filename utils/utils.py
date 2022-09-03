
# import libraries
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from sklearn.model_selection import GridSearchCV, train_test_split


def import_data(pth) -> pd.DataFrame:
    """

    :param pth:
    :return:
    """
    try:
        df = pd.read_csv(pth)
        logging.info(f'INFO: Data has been imported')
        return df
    except FileNotFoundError:
        logging.error(f"ERROR: {FileNotFoundError}, unable to import data")


def split_data(df: pd.DataFrame, target: str, test_size: float,
               random_state: int):
    """

    :param df:
    :param target:
    :param test_size:
    :param random_state:
    :return:
    """

    try:
        df.select_dtype(exclude=['object'])
        X = df.drop(target, axis=1)
        Y = df[target]
        x_train, x_test, y_train, y_test, feature_names = \
            train_test_split(X, Y, test_size=test_size,
                             random_state=random_state), \
            df.drop(target, axis=1).columns

        logging.info(f"INFO: Data splitted")
        return (x_train, x_test, y_train, y_test), feature_names
    except KeyError:
        logging.error(f"{KeyError}, unable to split the data")


def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series, model, save_dir: Path,
                param_grid:dict):
    """

    :param model:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param save_dir:
    :param grid_search:
    :param probabilities:
    :return:
    """

    try:
        if grid_search:
            g_search = GridSearchCV(estimator=model,
                                    param_grid=param_grid,
                                    cv=cv, n_jobs=n_jobs, verbose=verbose)
            model_fit = g_search.fit(X_train, y_train)
        else:
            model_fit = model.fit(X_train, y_train)

        logging.info(f'INFO: {model_fit} has been fitted')

        if save_dir:
            save_model(save_dir, model)
            logging.info(f'INFO: Model saved: {model_fit}')

        predictions = model_fit.predict_probalities(X_test) \
            if probabilities else model_fit.predict(X_test)
        logging.info(f'INFO: Prediction made on test '
                     f'data using probability = {probabilities}')
        return model_fit, predictions

    except TypeError:
        logging.error(f'ERROR: {TypeError} Unable to fit the model')


def create_dir(path: Path):
    try:
        if path.is_dir() == False:
            logging.info(f'INFO: a folder has been created at the '
                         f'requested path {path}')


    except SyntaxError:
        logging.error('ERROR: while creating the folder in the '
                      'requested path {path}')


def save_model(path, model):
    try:
        create_dir(path)
        return joblib.dump(model, f'{path}/{type(model).__name__}.pkl')
    except Exception:
        logging.error(f'ERROR: model not dumped {Exception}')


def load_model(path):
    """
    Load .pkl model
    @param path:Path = path where the model is located
    @return model = model
    """
    try:
        model = joblib.load(path)
        logging.info("The model has been loaded")
        return model
    except (FileNotFoundError, MemoryError) as e:
        logging.error(f'{e} Unable to load the specified model')


logging.basicConfig(
    filename=Path(os.path.dirname(__file__), 'logs/project.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s ')
