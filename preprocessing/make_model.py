import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import numpy as np
from utils.utils import create_dir, logging, import_data, split_data, \
    train_model, corrplot

from omegaconf import DictConfig

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# %% Evaluate Object
class Evaluate:
    """
    Evaluate model
    """

    def __init__(self, config):
        super(Evaluate, self).__init__()

    def eval(self, model, train_data: pd.DataFrame, test_data: pd.DataFrame,
             labels, predictions, save_results_dir: Path):
        """
        :param model: 
        :param train_data: 
        :param test_data: 
        :param labels: 
        :param predictions: 
        :param save_results_dir: 
        :return: 
        """""
        try:
            if isinstance(model, GridSearchCV):
                model_name = type(model.best_estimator_).__name__
                coefficients = model.best_estimator_.feature_importances_
            else:
                model_name = type(model).__name__
                coefficients = model.coef_[0]

            create_dir(save_results_dir)

            self.get_classification_report(labels, predictions, model_name,
                                           save_results_dir)
            logging.info("INFO:  Classification report computed")

            self.get_roc(model, test_data, labels, model_name, save_results_dir)
            logging.info("INFO: ROC curve computed")

            self.get_shapley_values(model, test_data, save_results_dir,
                                    model_name, train_data)
            logging.info("INFO: Shapley values computed")

            self.features_importances(train_data.columns, coefficients,
                                      model_name, save_results_dir)
            logging.info("INFO: Feature importance")

        except Exception:
            logging.error(
                f'ERROR: Not able to compute evaluation: {Exception}')

    @staticmethod
    def get_classification_report(labels, predictions, model_name,
                                  save_results_dir):
        """
        Compute classification report
        :@param labels: ground truth labels
        :@param predictions: predictions made by the model
        :@param model_name: moddel name
        :@param save_results_dir: directory to save results
        :@returns: report

        """
        try:
            report = pd.DataFrame(
                classification_report(labels, predictions, output_dict=True))
            return report.transpose().to_csv(
                f'{save_results_dir}/{model_name}_classification_report.csv')
        except Exception:
            logging.error(f'ERROR: Classification report not computed \
            {Exception}')

    @staticmethod
    def get_roc(model, test_data, labels, model_name, save_dir):
        """

        :param model:
        :param test_data:
        :param labels:
        :param model_name:
        :param save_dir:
        :return:
        """
        try:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        except ValueError:
            logging.error(f'ERROR: ROC not computed {ValueError}')

    @staticmethod
    def get_shapley(model, train_data, test_data, save_dir, model_name):
        """
        Compute shapley values
        :param model:
        :param test_data:
        :param save_dir:
        :return:
        """

        try:
            explainer = None
            plt.figure(figsize=(20, 10))
            if isinstance(model, GridSearchCV):
                model = model.best_estimator_
            if model_name in ['LogisticRegression', 'LinearRegression']:
                masker = shap.maskers.Independent(data=train_data)
                explainer = shap.LinearExplainer(model, masker=masker)
            elif model_name in ['RandomForestClassifier']:
                explainer = shap.TreeExplainer(model)
            else:
                logging.info('Add your model type to tree, linear, '
                             'gradient or deep explainer '
                             'and add condition to this function')
            shap_values = explainer.shap_values(test_data)
            shap.summary_plot(shap_values, test_data, show=False)
            plt.savefig(f'{save_dir}/{type(model).__name__}'
                        f'_shapley_values.png', bbox_inches='tight')
        except (AssertionError, AttributeError, ValueError,
                TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during shapley values compute: {err}')

    @staticmethod
    def get_features_importances(feature_names: pd.Index,
                                 coefficients: np.array, model_name: str,
                                 output_dir: str):
        """
        Compute features importances and save them into a png file
        :param feature_names: column names used for training
        :param coefficients: coefficients of each column used in training which corresponds to their weight
        in the prediction
        :param model_name: name of the model used like 'LogisticRegression' or 'RandomForest'
        :param output_dir: directory where to store result
        :returns a png file with features importances computed
        """
        try:
            plt.figure(figsize=(20, 10))
            create_dir(output_dir)
            df = pd.DataFrame(
                zip(feature_names, coefficients),
                columns=['feature', 'coefficient']).sort_values(
                by=['coefficient'],
                ascending=False)
            # Plot Searborn bar chart
            sns.barplot(x=df['coefficient'],
                        y=df['feature'])
            # Add chart labels
            plt.title(model_name + 'FEATURE IMPORTANCE')
            plt.xlabel('FEATURE IMPORTANCE')
            plt.ylabel('FEATURE NAMES')
            plt.savefig(f'{output_dir}/{model_name}_features_importances.png',
                        bbox_inches='tight')
        except (ValueError, TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during features importance compute: {err}')


class EncoderHelper:

    def __init__(self, config):
        self.config = config

    def encode(self, data):
        try:
            data[self.config['DATA_INFO']['NEW_TARGET_COL']] = self.get_encoding(data, self.config['DATA_INFO'])
            data = data.drop(self.config['DATA_INFO']['TARGET_COL'], axis=1)
            logging.info(f'Target column has been encoded')

            data = self.get_categorical_mean_encoding(data, self.config['DATA_INFO'])
            logging.info(f'Categorical features encoded')
            return data

        except BaseException as e:
            logging.error(f'{e} Unable to encode the outcome variable')

    @staticmethod
    def get_encoding(data, data_info:dict):
        try:
            target = data_info['TARGET_COL']
            target_encoding = data_info['ENCODING_TARGET']
            enc_dict = {}
            for key, value in target_encoding.items():
                enc_dict = {key: data[target] == value}
            return np.select(enc_dict.values(), enc_dict.keys())
        except BaseException as e:
            logging.info(f'Unable to encode target feature {e}')

    @staticmethod
    def get_categorical_mean_encoding(data: pd.DataFrame, data_info: dict):
        """
        :param data:
        :param data_info:
        :return:
        """
        for feature in data_info['CATEGORICAL_COLS']:
            feature_groups = round(data.groupby(feature).mean()[data_info['NEW_TARGET_COL']], 2)
            data[f'{feature}_{data_info["NEW_TARGET_COL"]}'] = data[feature].map(feature_groups)
            data = data.drop(feature, axis=1)
        return data


class DataExploration:
    def __init__(self, config:DictConfig):
        self.config = config

    def eda(self, data):
        """

        :param data:
        :param save_dir:
        :return:
        """
        try:
            save_dir = Path(self.config['DIRECTORIES']['PROJECT_DIR'], self.config['DIRECTORIES']['RESULTS_DIR'], 'EDA')
            create_dir(save_dir)
            logging.info(f'Starting EDA analysis. Saving Results in {save_dir}')

            # Unavariate analysis
            self.univariate_exploration(data, self.config['DATA_INFO'], save_dir)
            self.bivariate_exploration(data, self.config['DATA_INFO'], save_dir)
            self.correlation_matrix(data, self.config['DATA_INFO'], save_dir)

        except BaseException as e:
            logging.error(f" Unable to perform EDA analysis {e}")

    @staticmethod
    def univariate_exploration(data: pd.DataFrame, data_info: dict, save_dir):
        try:
            for category in data_info:
                cat = 'CATEGORICAL_COLS'
                num = 'NUMERICAL_COLS'
                if (category == cat) or (category == num):
                    logging.info(f' exploration on {category}')
                    if category == cat:
                        plot = getattr(sns, 'histplot')
                        features = data_info[cat]
                        type = 'histogram'
                    else:
                        plot = getattr(sns, 'displot')
                        features = data_info[num]
                        type = 'distplot'

                    for feature in features:
                        plotting = plot(data[feature])
                        name = 'univariate_exploration ' + type + ' ' + feature + '.png'
                        plt.xticks(rotation=45)
                        plt.xlabel(feature)
                        plotting.figure.savefig(Path(save_dir, name))
                        plt.close()
                        logging.info(f' univarite exploration computed and saved in {feature}')

        except BaseException as e:
            logging.error(f' {e} Unable to compute and save univariate analysis')

    @staticmethod
    def bivariate_exploration(data: pd.DataFrame, data_info: dict, save_dir: Path):
        try:
            for category in data_info:
                cat = 'CATEGORICAL_COLS'
                num = 'NUMERICAL_COLS'
                target = data_info['TARGET_COL']
                if (category == cat) or (category == num):
                    logging.info(f' exploration on {category}')
                    if category == cat:
                        plot = getattr(sns, 'catplot')
                        features = data_info[cat]
                        type = 'catplot'
                    else:
                        plot = getattr(sns, 'histplot')
                        features = data_info[num]
                        type = 'histplot'

                    for feature in features:
                        if type == 'histplot':
                            plotting = plot(data=data, x=feature, hue=target)
                        else:
                            plotting = plot(data=data, x=feature, hue=target, kind='count')
                        name = 'bivariate_exploration ' + type + ' ' + feature + '.png'
                        plt.xticks(rotation=45)
                        plotting.figure.savefig(Path(save_dir, name))
                        plt.close()
                        logging.info(f' bivariate exploration computed and saved in {feature}')

        except BaseException as e:
            logging.error(f' {e} Unable to compute and save bivariate analysis')


    @staticmethod
    def correlation_matrix(data:pd.DataFrame, data_info:dict, save_dir:Path):
        try:

            # Beutiful correlation plot taken from:
            # https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
            corr = data.corr()
            fig = plt.figure(figsize=(10, 10))
            corrplot(corr)
            fig.savefig(Path(save_dir, 'correlation_matrix' + '.png'))

        except BaseException as e:
            logging.error(f'{e} unable to compute correlation matrix')



class CustomerChurn(DataExploration, EncoderHelper, Evaluate):

    def __init__(self, config: DictConfig):
        DataExploration.__init__(self, config)
        EncoderHelper.__init__(self, config)
        Evaluate.__init__(self, config)
        self.config = config

    def predict(self):
        try:
            project_dir = self.config['DIRECTORIES']['PROJECT_DIR']
            data_dir = Path(project_dir, self.config['DIRECTORIES']['DATA_DIR'], 'bank_data.csv')
            df = import_data(data_dir)

            # %% Preprocessing
            self.eda(df)
            data = self.encode(df)
            (x_train, x_test, y_train, y_test), feature_names = \
                split_data(data, self.config['DATA_INFO']['NEW_TARGET_COL'], self.config['TEST_SIZE'],
                           self.config['RANDOM_STATE'])

            ## Init Models
            lcr = LogisticRegression(max_iter=self.config['LOGISTIC_REGRESSION']['MAX_ITER'])
            rfc = RandomForestClassifier(random_state=self.config['RANDOM_STATE'])

            save_model_dir = Path(project_dir, self.config['DIRECTORIES']['MODEL_DIR'])

            ## Train
            for model in [lcr, rfc]:
                if model == lcr:
                    model_fitted, predictions = train_model(x_train, x_test, y_train, y_test, model, param_grid=None,
                                                            save_dir=save_model_dir)
                elif model == rfc:
                    model_fitted, predictions = train_model(x_train, x_test, y_train, y_test, model,
                                                            param_grid=self.config['PARAM_GRID'], save_dir=save_model_dir)
                logging.info(f'INFO: Models trained')

                # Evaluate
                save_results_dir = Path(project_dir, self.config['DIRECTORIES']['RESULTS_DIR'])
                self.evaluate(model_fitted, x_train, x_test, y_test,
                              predictions, save_results_dir)

                logging.info(f'INFO: Model evaluated')
        except Exception:
            logging.error(f'ERROR: Error in pipeline CustomerChurn')
