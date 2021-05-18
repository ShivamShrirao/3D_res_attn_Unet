from tensorflow import tf
from tensorflow.keras import layers

from .conv_blocks import ConvNorm, NormAct, AttnBottleneckBlock


def down_stack(x, filters, blocks, strides=1, **kwargs):
    x = AttnBottleneckBlock(filters, strides=strides, **kwargs)(x)

    for i in range(1, blocks):
        x = AttnBottleneckBlock(filters, **kwargs)(x)

    return x

def up_stack(x, skip, filters, blocks, strides=1, **kwargs):
    if strides > 1:
        x = layers.UpSampling3D(data_format="channels_first")(x)

    x = layers.Concatenate(axis=1)([x, skip])
    for i in range(blocks):
        x = AttnBottleneckBlock(filters, **kwargs)(x)
    return x


def enc_dec(x, stack_args):     # x(64,64,96,80)
    x1 = down_stack(x ,  64, blocks=2, strides=2, **stack_args) # ( 256,32,48,40)
    x2 = down_stack(x1, 128, blocks=2, strides=2, **stack_args) # ( 512,16,24,20)
    x3 = down_stack(x2, 256, blocks=3, strides=2, **stack_args) # (1024, 8,12,10)
    # do norm act
    y = up_stack(x3,x2, 256, blocks=3, strides=2, **stack_args) # (1024, 8,12,10)
    y = up_stack(y, x1, 128, blocks=2, strides=2, **stack_args) # ( 512,16,24,20)
    y = up_stack(y, x ,  64, blocks=2, strides=2, **stack_args) # ( 256,32,48,40)
    return y


def build_network(cfg):
    stack_args = {'activation': cfg['activation'], 'groups': cfg['groups'], 'norm': cfg['norm'], 'dv': cfg['dv'],
                  'dp_rate': cfg['dp_rate'], 'dropout_type': cfg['dropout_type']}

    inp = layers.Input(shape=(4,depth,height,width))        # ( 4,128,192,160)
    x = inp
    x = ConvNorm(32, kernel_size=3, strides=2, activation=cfg['activation'], norm=cfg['norm'])(x)   # (32,64,96,80)
    x = ConvNorm(64, kernel_size=3, activation=cfg['activation'], norm=cfg['norm'])(x)  # (64,64,96,80)

    x = enc_dec(x, stack_args)                                                          # ( 256,32,48,40)

    x = layers.UpSampling3D(data_format="channels_first")(x)                            # (32, 64, 96, 80)
    x = ConvNorm(16, kernel_size=3, activation=cfg['activation'], norm=cfg['norm'])(x)  # (16,128,192,160)
    x = ConvNorm(3, kernel_size=3, activation=cfg['activation'], norm=cfg['norm'])(x)   # ( 3,128,192,160)
    x = layers.Softmax(axis=1)(x)         # softmax cause each pixel has unique class, no overlap with other classes, verified.

    return tf.keras.Model(inputs=inp, outputs=x)