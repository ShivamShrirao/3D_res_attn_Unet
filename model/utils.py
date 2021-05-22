import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

import numpy as np
import imageio

unique_name_gen = tf.Graph()         # just for some custom naming to get unique names.
get_name = lambda x: unique_name_gen.unique_name(x)

class CustomLayer(layers.Layer):
    def save_inits(self, locs):
        self.l_config = locs
        self.l_config.pop('self')
        self.l_config.pop('kwargs')
        self.l_config.pop('__class__')
        self._name = get_name(self.name)

    def get_config(self):
        return {**super().get_config(), **self.l_config}

class CustomCLR(tfa.optimizers.Triangular2CyclicalLearningRate):
    def __init__(self, offset=0, **kwargs):
        self.offset = offset
        super().__init__(**kwargs)

    def __call__(self, step):
        return super().__call__(step + self.offset)

    def get_config(self):
        return {**super().get_config(), 'offset': self.offset}


def compute_hcf(x, y):
    while y:
        x, y = y, x % y
    return x

def compute_factors(x, y, lim=32):      # get the highest common factor less than equal to `lim`
    hcf = compute_hcf(x, y)
    if isinstance(lim, str):            # max groups
        return hcf
    if hcf > lim:
        for i in range(hcf, 0, -1):
            if x%i == y%i == 0:
                if i <=lim:
                    return i
    else:
        return hcf

def get_gif(img, lbl, pred, fname, alpha = 0.5):  # [C, D, H, W]
    img = img * alpha
    img = np.stack((img,)*3, axis=-1)
    lbl = lbl.transpose(1,2,3,0)            # [D, H, W, C]
    pred = pred.transpose(1,2,3,0)          # [D, H, W, C]
    lbl[...,0] = 0
    pred[...,0] = 0
    with imageio.get_writer(fname, mode='I', fps=10) as writer:
        t_images = (img + lbl*(1-alpha)).astype(np.uint8)
        p_images = (img + pred*(1-alpha)).astype(np.uint8)
        o = np.stack([t_images, p_images], axis=2)
        o = o.reshape(*o.shape[:2], -1, o.shape[-1])
        for i in o:
            writer.append_data(i)