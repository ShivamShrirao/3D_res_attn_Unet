import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import SimpleITK as sitk


def preprocess_label(lbl):
    empty = lbl == 0
    ncr = lbl == 1
    ed = lbl == 2
    et = lbl == 4
    lbl = tf.stack([empty, ncr, ed, et])
    return tf.cast(lbl, tf.float32)

def read_img(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    scale = tf.reduce_max(img)/2
    img = (img/scale) - 1              # -1 to 1
    return tf.cast(img, tf.float32)

def load_img(path_list):
    img = tf.stack([read_img(path) for path in path_list[:-1]])
    lbl = sitk.GetArrayFromImage(sitk.ReadImage(path_list[-1]))
    lbl = tf.convert_to_tensor(lbl, dtype=tf.uint8)
    lbl = preprocess_label(lbl)
    return img, lbl         # [B, C, D, H, W]

def load_paths(paths):
    img, lbl = load_img([x.decode() for x in paths.numpy()])
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