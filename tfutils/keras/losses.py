import tensorflow as tf

def triplet_loss(dim, alpha=0.2):
    def loss(y_true, y_pred):
        # Assumes a shape of (batch_size, embedding_size*3)
        anchor, positive, negative = y_pred[:,:dim], y_pred[:,dim:2*dim], y_pred[:,2*dim:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        loss = positive_dist - negative_dist + alpha
        return tf.maximum(0., loss)
    return loss