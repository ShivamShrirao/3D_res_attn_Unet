import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import SimpleITK as sitk


def preprocess_label(lbl):
    empty = lbl == 0
    liv = lbl == 1
    les = lbl == 2
    lbl = tf.stack([empty, liv, les])
    return tf.cast(lbl, tf.float32)

def read_img(path):
    img = np.load(path)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img

def load_img(path_list):
    img = read_img(path_list[0])[None,...]
    lbl = read_img(path_list[-1])
    lbl = preprocess_label(lbl)
    return img, lbl         # [C, D, H, W]

def load_paths(paths):
    img, lbl = load_img([x.decode() for x in paths.numpy()])
    # img, lbl = tf.cast(img, dtype=tf.float32), tf.cast(lbl, dtype=tf.float32)
    return img, lbl

load_paths_wrapper = lambda x: tf.py_function(load_paths, [x], (tf.float32, tf.float32))

# only does in X axis
def random_rotate3D(img, lbl):          # img [C, D, H, W]
    if tf.random.uniform([], 0, 1, dtype=tf.float32) <= 0.2:
        img = tf.transpose(img, perm=[1,2,3,0])     # make channel last
        lbl = tf.transpose(lbl, perm=[1,2,3,0])
        angle = tf.random.uniform([], 0, 7, dtype=tf.float32)
        img = tfa.image.rotate(img, angle, interpolation='nearest', fill_mode='constant', fill_value=0)
        lbl = tfa.image.rotate(lbl, angle, interpolation='nearest', fill_mode='constant', fill_value=0)
        img = tf.transpose(img, perm=[3,0,1,2])     # make channel first
        lbl = tf.transpose(lbl, perm=[3,0,1,2])
    return img, lbl

def random_flip3D(imgs, lbls):
    for _ in range(tf.random.uniform([], 0 , 2, dtype=tf.int32)):
        dim = tf.random.uniform([], 2, 5, dtype=tf.int32)
        imgs = tf.reverse(imgs, [dim])
        lbls = tf.reverse(lbls, [dim])
    return imgs, lbls