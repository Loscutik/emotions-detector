import io

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf


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