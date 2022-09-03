import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import numpy as np
from utils.utils import create_dir, logging, import_data, split_data, \
    train_model
from omegaconf import DictConfig

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#%% Evaluate Object
class Evaluate:
    """
    Evaluate model
    """

    def __init__(self):
        super(Evaluate, self).__init__()

    def eval(self, model, train_data: pd.DataFrame, test_data: pd.DataFrame,
             labels, predictions, save_results_dir: Path):
        """
        Function that calls all actions for model evaluation
        :@param model: skelearn objec
        :@param train_data: data used for trainning
        :@param test_data: data for testing
        :@param labels: ground truth labels
        :@param predictions: predictions made by the model
        :@param save_results_dir: directory to save results
        :@return : figures model evaluation
        """
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
            plt.figure(figsize=(10,8))
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
    """
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    """
    def __init__(self):
        pass

class DataExploration:
    def __init__(self):
        pass

class CustomerChurn(DataExploration, EncoderHelper, Evaluate):

    def __init__(self, config:DictConfig):
        DataExploration.__init__(self, config)
        DataExploration.__init__(self, config)
        EncoderHelper.__init__(self, config)
        Evaluate.__init__(self, config)
        self.config = config

    def predict(self):
        try:
            project_dir = self.config['DIRECTORIES']['PROJECT_DIR']
            DATA_PATH = Path(project_dir, self.config['DIRECTORY']['DATA_DIR'])
            df = import_data(DATA_PATH)

            # %% Preprocessing
            self.eda(df, self.config)
            self.encoder(df, self.config['TARGET_COL'])
            (x_train, x_test, y_train, y_test), feature_names = \
                split_data(df, 'Churn', self.config['TEST_SIZE'],
                           self.config['RANDOM_STATE'])

            ## Init Models
            lcr = LogisticRegression(max_iter=self.config['LOGISTIC_REGRESSION']
            ['MAX_ITER'])
            rfc = RandomForestClassifier(random_state=self.config['RANDOM_STATE'])

            save_model_dir = Path(project_dir,
                                  self.config['DIRECTORIES']['MODEL_DIR'])

            ## Train
            for model in [lcr, rfc]:
                if model == lcr:
                    model_fitted, predictions = train_model(x_train,
                                                            x_test, y_train,
                                                            y_test ,model,
                                                            save_dir=
                                                            save_model_dir)
                elif model == rfc:
                    PARAM = self.config['PARAM_GRID']
                    model_fitted, predictions = train_model(x_train, x_test,
                                                            y_train, y_test,
                                                            model,
                                                            save_dir=
                                                            save_model_dir,
                                                            param_grid =
                                                            PARAM)
                logging.info(f'INFO: Models trained')

                #Evaluate
                save_results_dir = Path(project_dir,
                                  self.config['DIRECTORIES']['RESULTS_DIR'])
                self.evaluate(model_fitted, x_train, x_test, y_test,
                              predictions, save_results_dir)

                logging.info(f'INFO: Model evaluated')
        except Exception:
            logging.error(f'ERROR: Error in pipeline CustomerChurn')
