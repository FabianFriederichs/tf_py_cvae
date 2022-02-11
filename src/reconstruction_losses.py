import tensorflow as tf
import numpy as np

### Reconstruction losses suitable for use with the (C)VAE implementations

## functions implementing negative log likelihood of different distributions
#  (default axis parameters assumes 0th axis is batch dimension, functions return a vector of size batch)

# n-D gaussian distribution with covariance c * I
def nll_gaussian(x, mean, dim_axis = 1, c = 1.0):
    return tf.math.reduce_sum(0.5 * tf.math.square(x - mean) / c, axis = dim_axis)

# n-D independent bernoulli distribution
def nll_bernoulli(x, p, dim_axis = 1):
    return tf.math.reduce_sum(-x * tf.math.log(p) - (1.0 - x) * tf.math.log(1.0 - p), axis = dim_axis)

## losses directly usable with keras
# Scaled standard gaussian loss
class NllScaledStandardGaussianLoss(tf.keras.losses.Loss):
    def __init__(self, c = 1.0):
        super(NllScaledStandardGaussianLoss, self).__init__()
        self.c = c
    # computes negative log likelihood loss for each sample in batch and returns the mean
    def call(self, y_true, y_pred):
        x = tf.convert_to_tensor(value = y_true)
        p = tf.convert_to_tensor(value = y_pred)
        return tf.reduce_mean(nll_gaussian(x, p, range(1, p.shape.rank), self.c), axis = 0)

# Scaled standard gaussian loss
class NllStandardGaussianLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(NllStandardGaussianLoss, self).__init__()
    # computes negative log likelihood loss for each sample in batch and returns the mean
    def call(self, y_true, y_pred):
        x = tf.convert_to_tensor(value = y_true)
        p = tf.convert_to_tensor(value = y_pred)
        return tf.reduce_mean(nll_gaussian(x, p, range(1, p.shape.rank), 1.0), axis = 0)

# Bernoulli loss
class NllBernoulliLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits = True):
        super(NllBernoulliLoss, self).__init__()
        self.from_logits = from_logits
    # computes negative log likelihood loss for each sample in batch and returns the mean
    def call(self, y_true, y_pred):
        x = tf.convert_to_tensor(value = y_true)
        if self.from_logits:
            p = tf.math.sigmoid(tf.convert_to_tensor(value = y_pred))
        else:
            p = tf.convert_to_tensor(value = y_pred)
        return tf.reduce_mean(nll_bernoulli(x, p, range(1, p.shape.rank)), axis = 0)