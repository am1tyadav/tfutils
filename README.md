# TF Utils

A module with a bunch of utilites for TensorFlow + Keras. These are some functions and classes that I found myself writing more than once while creating various projects. Hopefully, some of it is useful for others as well.

## Installation and Usage

Install:

`pip3 install git+https://github.com/am1tyadav/tfutils.git`

Usage:

```python
import tfutils

model.fit(
    x, y,
    epochs=1,
    callbacks=[
        tfutils.keras.callbacks.SimpleTrainingPlot(plt)
    ]
)
```

What's available:

## Keras Callbacks

|Utility|Description|
|---|---|
|`tfutils.keras.callbacks.SimpleTrainingPlot(plt)`|Requires `matplotlib.pyplot` argument passed as `plt`, and returns a Keras callback that will plot training metrics `loss, val_loss, accuracy, val_accuracy`. Please note the full keyword `accuracy` is used and not `acc`.|
|`tfutils.keras.callbacks.PlotEmbedding(plt, embedding_model, x_test, y_test)`|Requires `matplotlib.pyplot` argument passed as `plt`, and returns a Keras callback that will plot a 2-dimensional representation of embedding from the `embedding_model` on the set `x_test`, colored with the values of labels `y_test`.|

## Keras Losses

|Utility|Description|
|---|---|
|`tfutils.keras.losses.triplet_loss(dim, alpha)`|Returns a Keras Loss function. Argument `dim` is the dimension of embedding, and `alpha` is the margin with default value set to `0.2`|

## Keras Plotting

|Utility|Description|
|---|---|
|`tfutils.keras.plotting.plot_training_history(plt, history)`|Plots training history from the history object returned from Keras' `model.fit()` to the `plt` object passed as argument, and then returns the `plt` object.|

## Datasets

|Utility|Description|
|---|---|
|`tfutils.datasets.mnist.load_data()`|Returns the MNIST dataset after normalizing, and reshaping the examples, and one hot encoding the labels for both training and test sets.|
|`tfutils.datasets.mnist.plot_ten_random_examples(plt, x, y, p=None)`|Plots ten random examples from MNIST examples and labels `x` and `y` to the `matplotlib.pyplot` passed as `plt`, and returns the `plt` object. Optinally predictions can be passed as `p`.|
