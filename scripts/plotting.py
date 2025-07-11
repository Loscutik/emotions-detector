import io

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LearningCurveDisplay
import tensorflow as tf
from helpers import is_iterable


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(7, 6))
    # Compute the labels from the normalized confusion matrix.
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_norm, annot=True, cmap=plt.cm.Blues, fmt='.3g', xticklabels=class_names,
                     yticklabels=class_names, square=True)
    # plt.axis.XAxis.tick_top()
    plt.tick_params(axis='x', bottom=False, labelbottom=False, labeltop=True, rotation=55)
    plt.tick_params(axis='y', left=False, rotation=0)

    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Predicted Label', loc='left')
    ax.set_ylabel('True Label', loc='top')
    ax.set_title("Confusion matrix", loc='center', fontweight='bold')

    # plt.tight_layout()
    return figure


def plot_learning_curves(X, y, estimators, ncols=3):
    count_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),  #it is the default value
        "cv": 10,
        "scoring": "roc_auc_ovr",
        "n_jobs": -1,
    }
    plot_params = {
        "line_kw": {"marker": "o"},
        #default=”fill_between” -The style used to display the score standard deviation around the mean score. If None, no representation of the standard deviation is displayed.
        "std_display_style": "fill_between",
        # The name of the score used to decorate the y-axis of the plot. It will override the name inferred from the scoring parameter.
        "score_name": "Score: ROC_AUC_ovr",
    }

    def plot(estimator, ax=None):
        # score_type is the type of score to plot. Can be one of "test", "train", or "both".
        LearningCurveDisplay.from_estimator(estimator, **count_params, score_type="both", **plot_params, ax=ax)
        if ax is None:
            plt.title(f'Learning curve of {estimator.__class__.__name__}')
        else:
            ax.set_title(f'Learning curve of {estimator.__class__.__name__}')

    if not is_iterable(estimators):
        plot(estimators)
    elif len(estimators) == 1:
        plot(estimators[0])
    else:
        nrows = len(estimators) // ncols
        if len(estimators) % ncols != 0:
            nrows += 1

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 5 * nrows), sharey=True)

        if nrows == 1:
            for idx, estimator in enumerate(estimators):
                plot(estimator,ax=ax[idx])
        else:
            for idx, estimator in enumerate(estimators):
                i = idx // ncols
                j = idx % ncols
                plot(estimator,ax=ax[i, j])


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a tensor image and
  returns it. The supplied figure is closed and inaccessible after this call.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4) #4: output an RGBA image.
  # Add the batch dimension, because summary.image needs pixel data with shape [k, h, w, c],
  image = tf.expand_dims(image, 0)
  return image