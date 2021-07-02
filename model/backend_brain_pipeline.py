import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os

from .conv_blocks import *
from .losses import *
from .utils import *

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

depth = 128 # 155
height = 192 # 240
width = 160 # 240
channel_names = ['_t1.', '_t2.', '_t1ce.', '_flair.']   # do not change order
# T1-weighted (T1), post-contrast T1-weighted (T1ce), T2-weighted (T2), and T2 Fluid Attenuated Inversion Recovery (FLAIR)
out_channels = ['empty', 'ncr', 'ed', 'et']
# necrotic and non-enhancing tumor core (NCR), peritumoral edema (ED), GD-enhancing tumor(ET)

# ET : available
# TC : ET + NCR
# WT : ET + NCR + ED

# model.save("model-best.h5", include_optimizer=False)

model = tf.keras.models.load_model("model-best.h5", custom_objects={'ConvNorm': ConvNorm,
                                                                    'NormAct': NormAct,
                                                                    'AttnBottleneckBlock': AttnBottleneckBlock,
                                                                    'BasicBlock': BasicBlock,
                                                                    'InvertedResBlock': InvertedResBlock,
                                                                    'SqueezeExcite': SqueezeExcite,
                                                                    'MHSA3D': MHSA3D,
                                                                    'AbsPosEmb': AbsPosEmb,
                                                                    'dsc': dsc,
                                                                    'FocalTversky': FocalTversky,
                                                                    'CustomCLR': CustomCLR,
                                                                    }, compile=False)

def sort_path_list(path_list):
    ret = []
    for cnl in channel_names:
        for p in path_list:
            if cnl in os.path.split(p)[-1]:
                ret.append(p)
                path_list.remove(p)
                break
        else:
            print(f"[!] Channel {cnl} not found.")
    return ret

def read_img(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    scale = tf.reduce_max(img)/2
    img = (img/scale) - 1              # -1 to 1
    return tf.cast(img, tf.float32)

def load_img(path_list):
    assert len(path_list)==4, f"There must be 4 channel paths, got {len(path_list)} instead."
    path_list = sort_path_list(path_list)
    img = tf.stack([read_img(path) for path in path_list])
    return img         # [C, D, H, W]

def final_augmentation(imgs):       # input imgs[B, 4, 155, 240, 240], output[B, 4, 128, 192, 160]
    imgs = center_crop3D(imgs)
    # imgs = tf.image.per_image_standardization(imgs)   # source code checked, it's fine for 3D
    # imgs = (imgs - mean) / std
    return imgs

def center_crop3D(imgs, train=True):          # Crops image to size (128, 192, 160)
    d_cr, h_cr, w_cr = 13, 24, 40
    imgs = imgs[:, :, d_cr:d_cr+depth, h_cr:h_cr+height, w_cr:w_cr+width]
    return imgs

def make_gif(img, pred, fname, alpha = 0.5):  # [C, D, H, W]
    img = img * alpha
    img = np.stack((img,)*3, axis=-1)
    pred = pred.transpose(1,2,3,0)          # [D, H, W, C]
    with imageio.get_writer(fname, mode='I', fps=10) as writer:
        p_images = (img + pred*(1-alpha)).astype(np.uint8)
        for i in p_images:
            writer.append_data(i)

def process_pipeline(paths, fname="out.gif"):
    imgs = load_img(paths.copy())[None,:]      # add batch dimension
    imgs = final_augmentation(imgs)
    preds = model(imgs)[0,1:]._numpy()
    img = imgs[0,1]._numpy()
    mn = img.min()
    mx = img.max()
    img = (img - mn)/(mx - mn) * 255
    make_gif(img, preds*255, fname=fname)