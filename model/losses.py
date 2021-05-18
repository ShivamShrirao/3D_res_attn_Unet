import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import backend as K

def label_smooth(y_true):
    num_classes = tf.cast(tf.shape(y_true)[1], y_true.dtype)
    label_smoothing = 0.1
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

def cce(y_true, y_pred):
    # y_true = label_smooth(y_true)
    return K.categorical_crossentropy(y_true, y_pred, axis=1)


# This func is just for metric, First channel is empty so is ignored in dice coefficient calculation.
def dsc(y_true, y_pred):     # input is (B,C,H,W,D)
    y_true = tf.cast(y_true[:,1:], tf.float32)
    y_pred = tf.cast(y_pred[:,1:], tf.float32)
    C = tf.shape(y_true)[1]
    y_true_T = tf.transpose(y_true, (1,0,2,3,4))    # (C, B,H,W,D)
    y_pred_T = tf.transpose(y_pred, (1,0,2,3,4))    # (C, B,H,W,D)
    y_true_f = tf.reshape(y_true_T, (C, -1))        # (C, B*H*W*D)
    y_pred_f = tf.reshape(y_pred_T, (C, -1))        # (C, B*H*W*D)

    intersection = tf.reduce_sum(y_true_f * y_pred_f, -1)
    denom = tf.reduce_sum(y_true_f, -1) + tf.reduce_sum(y_pred_f, -1)

    smooth = 1.0
    score = (2.0 * intersection + smooth) / (denom + smooth)
    return score

class FocalTversky(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def call(self, y_true, y_pred):     # input is (B,C,H,W,D)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        C = tf.shape(y_true)[1]
        y_true_T = tf.transpose(y_true, (1,0,2,3,4))    # (C, B,H,W,D)
        y_pred_T = tf.transpose(y_pred, (1,0,2,3,4))    # (C, B,H,W,D)
        y_true_f = tf.reshape(y_true_T, (C, -1))        # (C, B*H*W*D)
        y_pred_f = tf.reshape(y_pred_T, (C, -1))        # (C, B*H*W*D)

        true_pos = tf.reduce_sum(y_true_f * y_pred_f, -1)
        false_neg = tf.reduce_sum(y_true_f * (1-y_pred_f), -1)
        false_pos = tf.reduce_sum((1-y_true_f) * y_pred_f, -1)
        tvsk = (true_pos + self.smooth)/(true_pos + self.alpha*false_pos + self.beta*false_neg + self.smooth)
        tl = 1. - tvsk
        return tf.reduce_sum(tf.pow(tl, self.gamma))