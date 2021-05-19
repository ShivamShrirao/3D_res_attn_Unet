from tensorflow import tf
from tensorflow.keras import layers

from .conv_blocks import ConvNorm, NormAct, down_stack, up_stack


def enc_dec(x, frac_dv, stack_args):     # x(64,64,96,80)
    x1 = down_stack(x ,  64, blocks=2, strides=2, **stack_args)     # (256,32,48,40)
    x2 = down_stack(x1, 128, blocks=2, strides=2, **stack_args)     # (512,16,24,20)
    x3 = down_stack(x2, 256, blocks=2, strides=2, frac_dv=frac_dv, **stack_args) # (1024, 8,12,10)
    x4 = down_stack(x3, 256, blocks=3, strides=2, frac_dv=frac_dv, **stack_args) # (1024, 4, 6, 5)

    y = up_stack(x4,x3, 128, blocks=3, strides=2, frac_dv=frac_dv, **stack_args) # (1024+1024->512,8,12,10)
    y = up_stack(y, x2,  64, blocks=2, strides=2, **stack_args)     # (512+512->256,16,24,20)
    y = up_stack(y, x1,  32, blocks=2, strides=2, **stack_args)     # (256+256->128,32,48,40)
    y = up_stack(y, x ,  16, blocks=1, strides=2, **stack_args)     # (128+ 64-> 64,64,96,80)
    return y


def build_network(cfg, input_shape=(4,128,192,160), classes=4):
    stack_args = {'activation': cfg['activation'], 'groups': cfg['groups'], 'norm': cfg['norm'],
                  'dp_rate': cfg['dp_rate'], 'dropout_type': cfg['dropout_type']}

    inp = layers.Input(shape=input_shape)                           # ( 4,128,192,160)
    x = inp
    x = ConvNorm(32, kernel_size=3, strides=2, activation=cfg['activation'], norm=cfg['norm'])(x)   # (32,64,96,80)
    x = ConvNorm(64, kernel_size=3, do_norm_act=False)(x)                               # (64, 64, 96, 80)

    x = enc_dec(x, cfg['frac_dv'], stack_args)                                          # (64, 64, 96, 80)

    x = NormAct(activation=cfg['activation'], norm=cfg['norm'])(x)
    x = layers.UpSampling3D(data_format="channels_first")(x)                            # (64,128,192,160)
    x = ConvNorm(16, kernel_size=3, activation=cfg['activation'], norm=cfg['norm'])(x)  # (16,128,192,160)
    x = ConvNorm(classes,kernel_size=3, activation=cfg['activation'], norm=cfg['norm'])(x)#(3,128,192,160)
    x = layers.Softmax(axis=1)(x)         # softmax cause each pixel has unique class, no overlap with other classes, verified.

    return tf.keras.Model(inputs=inp, outputs=x)