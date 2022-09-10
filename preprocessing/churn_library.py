from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from utils.utils import create_dir, logging, import_data, split_data, \
    train_model, corrplot

from omegaconf import DictConfig

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class Evaluate:
    """
    Evaluate model
    """

    def __init__(self):
        super(Evaluate, self).__init__()

    def eval(self, model, train_data: pd.DataFrame, test_data: pd.DataFrame,
             labels, predictions, save_results_dir: Path):
        """
        :param model: model to be evaluated
        :param train_data: data to be trained
        :param test_data: testing data
        :param labels: ground truth labels
        :param predictions: model predictions
        :param save_results_dir: directory to save results
        :return: plots of features importance, eda, shapley
        """
        try:
            create_dir(save_results_dir)
            if isinstance(model, GridSearchCV):
                model_name = type(model.best_estimator_).__name__
                coefficients = model.best_estimator_.feature_importances_
            else:
                model_name = type(model).__name__
                coefficients = model.coef_[0]

            self.get_classification_report(labels, predictions, model_name,
                                           save_results_dir)
            self.get_roc(
                model,
                test_data,
                labels,
                model_name,
                save_results_dir)
            self.get_shapley(model, test_data, train_data, save_results_dir,
                             model_name)
            self.get_features_importances(train_data.columns, coefficients,
                                          model_name, save_results_dir)
        except BaseException as e:
            logging.error(f'{e} Not able to compute evaluation')

    @staticmethod
    def get_classification_report(labels, predictions, model_name,
                                  save_results_dir):
        """
        Funtion to comput and store a classification report
        :param labels: ground truth labels
        :param predictions: predictions made by the model
        :param model_name: moddel name
        :param save_results_dir:  directory to save results
        :return: report
        """

        try:
            report = pd.DataFrame(
                classification_report(labels, predictions, output_dict=True))
            report = report.transpose()
            report.to_csv(f'{save_results_dir}/'
                          f'{model_name}_classification_report.csv')
            logging.info(f'Clasification report computed '
                         f'and save for {model_name}')
        except BaseException as e:
            logging.error(f' {e} Classification report not computed')

    @staticmethod
    def get_roc(model, test_data, labels, model_name, save_dir):
        """
        Function to compute ROC
        :param model: model to compute ROC
        :param test_data: testing data
        :param labels: ground truth labels
        :param model_name: model name
        :param save_dir: directory to store data
        :return: compute ROC plot
        """
        try:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            plot_roc_curve(model, test_data, labels, ax=ax, alpha=0.8)
            plt.savefig(Path(save_dir, model_name + '_roc_curve.png'), dpi=300)
            logging.info(f'ROC curve computed for {model_name}')
        except ValueError as e:
            logging.error(f'{e} ROC not computed')

    @staticmethod
    def get_shapley(model, train_data, test_data, save_dir, model_name):
        """
        Compute shapley values
        :param model: model to compute shapley values
        :param test_data: testing dataset
        :param save_dir: directory to save data
        :return: plot shapley values
        """

        try:
            explainer = None
            plt.figure(figsize=(20, 10))
            if isinstance(model, GridSearchCV):
                model = model.best_estimator_

            if model_name in 'LogisticRegression':
                masker = shap.maskers.Independent(data=train_data)
                explainer = shap.LinearExplainer(model, masker=masker)

            elif model_name in 'RandomForestClassifier':
                explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(test_data)
            shap.summary_plot(shap_values, test_data, show=False)
            plt.savefig(Path(save_dir, type(model).__name__ +
                             '_shapley_values.png'))
            logging.info(f'Shapley values computed and saved for {model_name}')
        except (AssertionError, AttributeError, ValueError,
                TypeError, FileNotFoundError) as e:
            logging.error(f'{e} Unable to compute and plot Shapley values')

    @staticmethod
    def get_features_importances(feature_names: pd.Index, coefficients:
                                 np.array, model_name: str, save_dir):
        """
        Funtion to plot feature importances
        :param feature_names: name of the features
        :param coefficients: model coefficients
        :param model_name:  model name
        :param save_dir: directory to save plot
        :return: plot of feature importances
        """
        try:
            create_dir(save_dir)
            plt.figure(figsize=(20, 10))
            df = pd.DataFrame(
                zip(feature_names, coefficients),
                columns=['feature', 'coefficient']).sort_values(
                by=['coefficient'],
                ascending=False)
            # Plot Searborn bar chart
            sns.barplot(x=df['coefficient'],
                        y=df['feature'])
            # Add chart labels
            plt.title('Feature Importances ' + model_name)
            plt.xlabel('Importances')
            plt.ylabel('Names')
            plt.savefig(
                Path(
                    save_dir,
                    model_name +
                    '_features_importances.png'))
            logging.info(
                f'Feature importance computed and saved for {model_name}')
        except (ValueError, TypeError, FileNotFoundError) as e:
            logging.error(f'{e} unable to plot feature importnaces')


class EncoderHelper:
    """
    Categorical Encoder Object
    """

    def __init__(self):
        super(EncoderHelper, self).__init__()

    def encode(self, data):
        """

        :param data: experiment dataframe
        :return: encoded data frame
        """
        try:
            data[self.config['DATA_INFO']['NEW_TARGET_COL']] = \
                self.get_encoding(data, self.config['DATA_INFO'])
            data = data.drop(self.config['DATA_INFO']['TARGET_COL'], axis=1)
            logging.info(f'Target column has been encoded')

            data = self.get_categorical_mean_encoding(data,
                                                      self.config['DATA_INFO'])
            logging.info(f'Categorical features encoded')
            return data

        except BaseException as e:
            logging.error(f'{e} Unable to encode the outcome variable')

    @staticmethod
    def get_encoding(data, data_info: dict):
        """

        :param data: experiment dataframe
        :param data_info: dictionay of features names
        :return: target encoded
        """
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
        :param data: experiment data frame
        :param data_info: dictionary of feature names
        :return: dataframe with categorical features encoded
        """
        for feature in data_info['CATEGORICAL_COLS']:
            feature_groups = round(data.groupby(feature).mean()
                                   [data_info['NEW_TARGET_COL']], 2)
            data[f'{feature}_{data_info["NEW_TARGET_COL"]}'] = \
                data[feature].map(feature_groups)
            data = data.drop(feature, axis=1)
        return data


class DataExploration:
    """
    Data Exploration Object
    """

    def __init__(self):
        super(DataExploration, self).__init__()

    def eda(self, data):
        """

        :param data: dataframe to compute eda
        :param save_dir: directory to save results
        :return: plot of eda computations (categorical and numerical)
        """
        try:
            save_dir = Path(self.config['DIRECTORIES']['PROJECT_DIR'],
                            self.config['DIRECTORIES']['RESULTS_DIR'], 'EDA')
            create_dir(save_dir)
            logging.info(f'Starting EDA analysis. Saving '
                         f'Results in {save_dir}')

            # Unavariate analysis
            self.univariate_exploration(data, self.config['DATA_INFO'],
                                        save_dir)
            self.bivariate_exploration(data, self.config['DATA_INFO'],
                                       save_dir)
            self.correlation_matrix(data, self.config['DATA_INFO'],
                                    save_dir)

        except BaseException as e:
            logging.error(f" Unable to perform EDA analysis {e}")

    @staticmethod
    def univariate_exploration(data: pd.DataFrame, data_info: dict, save_dir):
        """
        Compute univariate exploration
        :param data: dataframe for exploration
        :param data_info: dictonay of feature names
        :param save_dir: dericetory to save results
        :return: plots of univariate exploration
        """
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
                        name = 'univariate_exploration ' + type + ' ' + \
                               feature + '.png'
                        plt.xticks(rotation=45)
                        plt.xlabel(feature)
                        plotting.figure.savefig(Path(save_dir, name))
                        plt.close()
                        logging.info(f' univarite exploration computed '
                                     f'and saved in {feature}')

        except BaseException as e:
            logging.error(f' {e} Unable to compute and save '
                          f'univariate analysis')

    @staticmethod
    def bivariate_exploration(data: pd.DataFrame, data_info: dict,
                              save_dir: Path):
        """

        :param data: dataframe for exploration
        :param data_info: dictonary of features names
        :param save_dir: directory to same results
        :return: plots with bivariate explorations
        """
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
                            plotting = plot(data=data, x=feature,
                                            hue=target, kind='count')
                        name = 'bivariate_exploration ' + type + ' ' + \
                               feature + '.png'
                        plt.xticks(rotation=45)
                        plotting.figure.savefig(Path(save_dir, name))
                        plt.close()
                        logging.info(f' bivariate exploration computed '
                                     f'and saved in {feature}')

        except BaseException as e:
            logging.error(f' {e} Unable to compute and save '
                          f'bivariate analysis')

    @staticmethod
    def correlation_matrix(
            data: pd.DataFrame,
            data_info: dict,
            save_dir: Path):
        """

        :param data: data for exploration
        :param data_info: dictionay with feature names
        :param save_dir: directory to save correlation plot
        :return: Correlation plot
        """
        try:

            # Beutiful correlation plot taken from:
            # https://towardsdatascience.com/better-heatmaps-and-
            # correlation-matrix-plots-in-python-41445d0f2bec
            corr = data.corr()
            fig = plt.figure(figsize=(10, 10))
            corrplot(corr)
            fig.savefig(Path(save_dir, 'correlation_matrix' + '.png'))

        except BaseException as e:
            logging.error(f'{e} unable to compute correlation matrix')


class CustomerChurn(DataExploration, EncoderHelper, Evaluate):
    """
    Customer Churn Pipeline Object
    """

    def __init__(self, config: DictConfig):
        DataExploration.__init__(self)
        EncoderHelper.__init__(self)
        Evaluate.__init__(self)
        self.config = config

    def predict(self):
        """
        Predict
        :return: predictions
        """
        try:
            project_dir = self.config['DIRECTORIES']['PROJECT_DIR']
            data_dir = Path(project_dir, self.config['DIRECTORIES']
                            ['DATA_DIR'], 'bank_data.csv')
            df = import_data(data_dir)

            # %% Preprocessing
            self.eda(df)
            data = self.encode(df)
            (x_train, x_test, y_train, y_test) = \
                split_data(data, self.config['DATA_INFO']['NEW_TARGET_COL'],
                           self.config['TEST_SIZE'],
                           self.config['RANDOM_STATE'])
            lcr = LogisticRegression(
                max_iter=self.config['LOGISTIC_REGRESSION']['MAX_ITER'])
            rfc = RandomForestClassifier(
                random_state=self.config['RANDOM_STATE'])
            save_model_dir = \
                Path(project_dir, self.config['DIRECTORIES']['MODEL_DIR'])

            for model in [lcr, rfc]:
                if model == lcr:
                    param = None
                elif model == rfc:
                    param = self.config['PARAM_GRID']

                model_fitted, predictions = train_model(
                    x_train, x_test, y_train, y_test, model, param_grid=param,
                    save_dir=save_model_dir)
                logging.info(f'Model {model} trained')
                save_results_dir = Path(
                    project_dir, self.config['DIRECTORIES']['RESULTS_DIR'])
                self.eval(model_fitted, x_train, x_test, y_test, predictions,
                          save_results_dir)
                logging.info(f'Model {model} evaluated')
        except BaseException as e:
            logging.error(f' {e} in pipeline CustomerChurn')
