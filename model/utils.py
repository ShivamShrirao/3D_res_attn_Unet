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


class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, warmup_steps, **kwargs):
        super().__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        return tf.cond(step < self.warmup_steps,
                       lambda: self.initial_learning_rate * (1-self.decay_rate) * (step / self.warmup_steps),
                       lambda: self.initial_learning_rate * self.decay_rate ** ((step-self.warmup_steps) / self.decay_steps))

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate,
                'decay_steps': self.decay_steps,
                'decay_rate': self.decay_rate,
                'warmup_steps': self.warmup_steps}


class WarmupExponentialDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, warmup_steps, restart_rate=0.4, **kwargs):
        super().__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.restart_rate = restart_rate
        self.r_steps = int(self.decay_steps * self.restart_rate)
    
    def __call__(self, step):
        step = tf.cast(step, tf.int32)
        restart = step % self.r_steps
        step, r_frac = tf.cond(step <= restart,
                               lambda: (step, tf.cast(1.0, tf.float64)),
                               lambda: (restart, self.restart_rate/tf.cast(step//self.r_steps, tf.float64)))
        wstp = tf.cast(self.warmup_steps*r_frac, tf.int32)             # fraction after restart
        return tf.cond(step < wstp,
                       lambda: self.initial_learning_rate*r_frac * (1-self.decay_rate) * (step / wstp),
                       lambda: self.initial_learning_rate*r_frac * self.decay_rate ** ((step-wstp) / self.decay_steps))

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate,
                'decay_steps': self.decay_steps,
                'decay_rate': self.decay_rate,
                'warmup_steps': self.warmup_steps,
                'restart_rate': self.restart_rate}


def compute_hcf(x, y):
    while y:
        x, y = y, x % y
    return x

def compute_factors(x, y, lim=48):      # get the highest common factor less than equal to `lim`
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
    with imageio.get_writer(fname, mode='I', fps=10) as writer:
        t_images = (img + lbl*(1-alpha)).astype(np.uint8)
        p_images = (img + pred*(1-alpha)).astype(np.uint8)
        o = np.stack([t_images, p_images], axis=2)
        o = o.reshape(*o.shape[:2], -1, o.shape[-1])
        for i in o:
            writer.append_data(i)