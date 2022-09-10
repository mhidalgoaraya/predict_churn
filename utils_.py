"""
utils_.py
Author Marco Hidalgo
"""
# import libraries
import os
from pathlib import Path
import logging
import logging.config
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import omegaconf


sns.set()


def import_data(file_dir):
    """

    :param file_dir: File directory
    :return: data imported
    """
    try:
        data = pd.read_csv(file_dir)
        logging.info('Data has been imported')
        return data
    except (FileNotFoundError, NameError) as err:
        logging.error( 'unable to import data. {}'.format(err))


def split_data(data: pd.DataFrame, target: str, test_size: float,
               random_state: int):
    """

    :param data: dataframe to split
    :param target: target to ssplit
    :param test_size: size for splitting
    :param random_state: for reproducibility
    :return: splited data
    """

    try:
        dependent_vars = data.drop(target, axis=1)
        independent_variable = data[target]
        x_train, x_test, y_train, y_test = train_test_split(
            dependent_vars, independent_variable, test_size=test_size,
            random_state=random_state,
            shuffle=True, stratify=independent_variable)

        logging.info("Data splitted")
        return x_train, x_test, y_train, y_test
    except KeyError as err:
        logging.error('unable to split the data {}'.format(err))


def train_model(x_train: pd.DataFrame, x_test: pd.DataFrame,
                y_train: pd.Series, model,
                save_dir, param_grid: dict = None):
    """

    :param x_test: Dependent variable for testing
    :param x_train: Dependent variable
    :param y_train: target data for training
    :param save_dir: savi directory
    :param model:model to train
    :return: model fittet
    """

    try:
        if param_grid is not None:
            if param_grid['compute']:
                g_search = GridSearchCV(
                    estimator=model,
                    param_grid=dict(param_grid['parameters']),
                    cv=param_grid['cv'], n_jobs=param_grid['n_jobs'],
                    verbose=param_grid['verbose'])
                model_fit = g_search.fit(x_train, y_train)
        else:
            model_fit = model.fit(y_train, y_train)

        logging.info('has been fitted {}'.format(model))

        if save_dir is not None:
            save_model(save_dir, model)
            logging.info('Model saved: {}'.format(model))

        if param_grid is not None:
            if param_grid['probabilities']:
                predictions = model_fit.predict_proba(x_test)
                logging.info('Prediction made on test data '
                             'using probability')
        else:
            predictions = model_fit.predict(x_test)
            logging.info('Prediction made on standard model fit')

        return model_fit, predictions

    except (TypeError, omegaconf.errors.ConfigKeyError, AttributeError) as err:
        logging.error(' Unable to fit the model {}'. format(err))


def create_dir(path: Path):
    """

    :param path: Directory to create directory
    :return: created directory
    """
    try:
        if path.is_dir() is False:
            os.makedirs(path)
            logging.info(' Directory created at the '
                         'requested path {}'.format(path))

    except SyntaxError as err:
        logging.error('During folder creation {}'.format(err))


def save_model(path, model):
    """

    :param path: Directory to save model
    :param model: model to store
    :return: saved model
    """
    try:
        create_dir(path)
        return joblib.dump(model, f'{path}/{type(model).__name__}.pkl')
    except TypeError as err:
        logging.error(f'model not dumped {err}')


def load_model(path):
    """

    :param path: Directory to load the model
    :return: model
    """

    try:
        model = joblib.load(path)
        logging.info("The model has been loaded")
        return model
    except (FileNotFoundError, MemoryError) as err:
        logging.error('Unable to load the specified model {}'.format(err))


# Configure logging
with open(Path(Path.cwd(), 'config', 'logging.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

logging.config.dictConfig(config)


# Taken from https://www.kaggle.com/code/drazen/heatmap-with-
# sized-markers/notebook
def heatmap(x, y, **kwargs):
    """

    :param x: data for corrrealtion
    :param y: taget variable
    :param kwargs: extra argurments
    :return: plot heatmap
    """
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1] * len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        # Range of values that will be mapped to the palette, i.e. min and max
        # possible correlation
        color_min, color_max = min(color), max(color)

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (
                color_max - color_min)  # position of
            # value in the input range, relative to the length of the input
            # range
            val_position = min(max(val_position, 0), 1)  # bound
            # the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1))  # target
            # index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1] * len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (
                size_max - size_min) + 0.01  # position of
            # value in the input range, relative to the length of the input
            # range
            val_position = min(max(val_position, 0), 1)  # bound the
            # position betwen 0 and 1
            return val_position * size_scale

    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(
        1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the left 14/15ths
    # of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k: v for k, v in kwargs.items() if k not in [
        'color', 'palette', 'color_range', 'size', 'size_range',
        'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k, v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45,
                       horizontalalignment='right')
    ax.set_yticks([v for k, v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost
        # column of the plot

        col_x = [0] * len(palette)  # Fixed x coordinate for the bars
        bar_y = np.linspace(color_min, color_max, n_colors)  #
        # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5] * len(palette),  # Make bars 5 units wide
            left=col_x,  # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets cro
        # p the plot somewhere in the middle
        ax.grid(False)  # Hide grid
        ax.set_facecolor('white')  # Make background white
        ax.set_xticks([])  # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vert
        # ical ticks for min, middle and max
        ax.yaxis.tick_right()  # Show vertical ticks on the right


def corrplot(data, size_scale=500, marker='s'):
    """
    Compute correlation plot
    :param data: data to compute correlation plot
    :param size_scale: scale
    :param marker: markers
    :return:
    """
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0, 1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )
