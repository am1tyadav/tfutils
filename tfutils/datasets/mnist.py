import tensorflow as tf
import numpy as np

def load_data(one_hot=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
    x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

def plot_ten_random_examples(plt, x, y, p=None):
    indices = np.random.choice(range(0, x.shape[0]), 10, replace=False)
    y = np.argmax(y, axis=1)
    if p is None:
        p = y
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(x[index].reshape((28, 28)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
        if y[index] == p[index]:
            col = 'g'
        else:
            col = 'r'
        plt.xlabel(str(p[index]), color=col)
    return plt

def load_subset(classes, x, y):
    """
    y should not be one hot encoded
    """
    x_subset = None
    for i, c in enumerate(classes):
        indices = np.squeeze(np.where(y == c))
        x_c = x[indices]
        if i == 0:
            x_subset = np.array(x_c)
        else:
            x_subset = np.concatenate([x_subset, x_c], axis=0)
    return x_subset
