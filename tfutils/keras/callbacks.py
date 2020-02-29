import tensorflow as tf
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class SimpleTrainingPlot(tf.keras.callbacks.Callback):
    """
    Requires matplotlib.pyplot passed as argument plt
    when this callback is instantiated.
    Training metric for accuracy needs to be set as 'accuracy'
    and not 'acc'
    """
    def __init__(self, plt):
        super(SimpleTrainingPlot, self).__init__()

        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1) # Losses
        self.ax2 = plt.subplot(1, 2, 2) # Accuracies
        plt.ion()
    
    def plot(self, epoch=None):
        if epoch is not None:
            self.ax1.clear()
            self.ax2.clear()

            self.ax1.plot(range(epoch), self.losses, label='Train')
            self.ax1.plot(range(epoch), self.val_losses, label='Val')
            self.ax1.set_xlabel('Epochs')
            self.ax1.set_ylabel('Loss')
            self.ax1.legend()

            self.ax2.plot(range(epoch), self.accs, label='Train')
            self.ax2.plot(range(epoch), self.val_accs, label='Val')
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Accuracy')
            self.ax2.legend()

            self.fig.canvas.draw()
    
    def on_train_begin(self, logs=None):
        self.val_accs = []
        self.accs = []
        self.val_losses = []
        self.losses = []
        
        self.fig.show()
        self.fig.canvas.draw()
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.accs.append(logs.get('accuracy'))
            self.val_accs.append(logs.get('val_accuracy'))
            self.plot(epoch+1)

class PlotEmbedding(tf.keras.callbacks.Callback):
    """
    Plot Embedding using the feature embedding generated from an
    embedding model passed in this callback's instance
    """
    def __init__(self, plt, embedding_model, x_test, y_test, use_tsne=True):
        super(PlotEmbedding, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.use_tsne = use_tsne
        self.fig = plt.figure()
        self.ax = plt.subplot(1, 1, 1)
        plt.ion()
    
    def plot(self, epoch=None):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        if self.use_tsne:
            out = TSNE(n_components=2).fit_transform(x_test_embeddings)
        else:
            out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax.clear()
        self.ax.scatter(out[:, 0], out[:, 1], c=self.y_test, cmap='seismic')
        self.fig.canvas.draw()
    
    def on_train_begin(self, logs=None):
        self.fig.show()
        self.fig.canvas.draw()
        self.plot()
        
    def on_epoch_end(self, epoch, logs=None):
        self.plot(epoch+1)
