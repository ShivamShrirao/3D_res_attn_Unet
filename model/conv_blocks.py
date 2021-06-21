import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from .utils import CustomLayer, compute_factors
from .attention import SqueezeExcite, MHSA3D, AbsPosEmb


def down_stack(x, filters, nblocks, block, strides=1, frac_dv=0, **kwargs):
    x = block(filters, frac_dv=frac_dv, strides=strides, **kwargs)(x)
    for i in range(1, nblocks):
        x = block(filters, frac_dv=frac_dv, **kwargs)(x)
    return x


def up_stack(x, skip, filters, nblocks, block, strides=1, frac_dv=0, **kwargs):
    if strides > 1:
        x = layers.UpSampling3D(data_format="channels_first")(x)
    x = layers.Concatenate(axis=1)([x, skip])

    for i in range(nblocks):
        x = block(filters, frac_dv=frac_dv, **kwargs)(x)
    return x


def hard_sigmoid(x):
    return tf.keras.activations.relu((x + 3.), max_value=6.) * (1. / 6.)

def hard_swish(x):
    return hard_sigmoid(x) * x


class NormAct(CustomLayer):
    def __init__(self, activation=tf.nn.leaky_relu, norm='gn', gn_grps=8, **kwargs):
        super().__init__(**kwargs)
        self.save_inits(locals())
        if norm == 'gn':
            # gn_grps = compute_factors(gn_grps, filters, gn_grps)
            self.norm = tfa.layers.GroupNormalization(groups=gn_grps, axis=1)
        else:
            self.norm = layers.BatchNormalization(axis=1)
        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        elif activation == 'hard_swish':
            activation =  hard_swish
        self.act = layers.Activation(activation)

    def call(self, inp, training=False):
        x = self.norm(inp, training=training)
        x = self.act(x)
        return x


class ConvNorm(CustomLayer):
    def __init__(self, filters, kernel_size=3, strides=1, groups=1, deconv=False, padding='same', use_bias=True,
                 activation=tf.nn.leaky_relu, do_norm_act=True, norm='gn', gn_grps=8, **kwargs):
        super().__init__(**kwargs)
        self.save_inits(locals())
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.groups = groups
        self.do_norm_act = do_norm_act
        self.activation = activation
        self.norm = norm
        self.gn_grps = gn_grps
        if deconv:
            print("[!] DeConv not ported yet.")

    def build(self, input_shape):
        in_filters = input_shape[1]
        self.lyrs = list()
        self.groups = compute_factors(in_filters, self.filters, self.groups) # just to make sure filters divisible by groups
        bias_state = (not self.do_norm_act) and self.use_bias
        self.lyrs.append(layers.Conv3D(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                       use_bias=bias_state , data_format="channels_first", groups=self.groups))
        if self.do_norm_act:
            self.lyrs.append(NormAct(activation=self.activation, norm=self.norm, gn_grps=self.gn_grps))

    def call(self, inp, training=False):
        x = inp
        for lyr in self.lyrs:
            x = lyr(x, training=training)
        return x


class AttnBottleneckBlock(CustomLayer):
    def __init__(self, filters, strides=1, activation=tf.nn.relu, expansion=4, dp_rate=0, dropout_type='Spatial',
                 groups=1, norm='gn', squeeze_attn=True, frac_dv=0, nheads=8, downsample_method="pool" , **kwargs):
        super().__init__(**kwargs)
        self.save_inits(locals())
        self.filters = filters
        self.expansion = expansion
        self.strides = strides
        self.out_filters = filters*expansion
        self.activation = activation
        self.groups = groups
        self.norm = norm
        self.squeeze_attn = squeeze_attn
        self.dropout = None
        self.dv = int(filters*frac_dv)
        self.conv_filters = filters - self.dv
        self.nheads = nheads
        self.downsample_method = downsample_method
        self.norm_act = NormAct(activation=activation, norm=norm)
        if dp_rate:
            if dropout_type == 'Spatial':
                self.dropout = layers.SpatialDropout3D(dp_rate, data_format="channels_first")
            else:
                self.dropout = layers.Dropout(dp_rate)

    def build(self, input_shape):
        self.shortcut = self.get_shortcut(input_shape)
        self.net = self.get_net(input_shape)

    def get_shortcut(self, input_shape):
        in_filters = input_shape[1]
        # operations for shortcut
        self.short_inp = True              # to decide input for shortcut, based on preact, filters and stride.
        shortcut = tf.keras.Sequential()

        if self.strides>1:
            shortcut.add(layers.AveragePooling3D(self.strides, data_format="channels_first"))
        if in_filters != self.out_filters:
            self.short_inp = False
            shortcut.add(ConvNorm(self.out_filters, kernel_size=1, strides=1, do_norm_act=False))
        return shortcut

    def get_net(self, input_shape):
        inp = layers.Input(shape=input_shape[1:])
        # Main bottleneck network
        x = ConvNorm(self.filters, kernel_size=1, activation=self.activation, norm=self.norm)(inp)

        conv_strides = self.strides
        c_x = x
        if self.strides > 1:
            x = layers.AveragePooling3D(self.strides, data_format="channels_first")(x)
            if self.downsample_method == "pool":
                conv_strides = 1
                c_x = x

        if self.conv_filters > 0:
            x_s = ConvNorm(self.conv_filters, kernel_size=3, strides=conv_strides, do_norm_act=False,
                           use_bias=False, groups=self.groups)(c_x)
        if self.dv > 0:
            x, attn = MHSA3D(dv=self.dv, nheads=self.nheads)(x)
            x_s = layers.Concatenate(axis=1)([x, x_s]) if self.conv_filters > 0 else x

        x = NormAct(activation=self.activation, norm=self.norm)(x_s)

        if self.squeeze_attn and self.dv==0:
            x = SqueezeExcite()(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = ConvNorm(self.out_filters, kernel_size=1, do_norm_act=False)(x)
        return tf.keras.Model(inputs=inp, outputs=x)

    def call(self, inp, training=False):
        x = inp
        x = self.norm_act(x, training=training)    # pre-act

        identity = self.shortcut(inp if self.short_inp else x, training=training)

        x = self.net(x, training=training)
        x = x + identity
        return x


class BasicBlock(AttnBottleneckBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_filters = self.filters
    
    def get_net(self, input_shape):
        inp = layers.Input(shape=input_shape[1:])
        x = inp

        conv_strides = self.strides
        if self.strides>1 and self.downsample_method == "pool":
            x = layers.AveragePooling3D(self.strides, data_format="channels_first")(x)
            conv_strides = 1
        x = ConvNorm(self.filters, kernel_size=3, strides=conv_strides, activation=self.activation, norm=self.norm,
                     groups=self.groups)(x)
        if self.squeeze_attn:
            x = SqueezeExcite()(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = ConvNorm(self.out_filters, kernel_size=3, do_norm_act=False)(x)

        return tf.keras.Model(inputs=inp, outputs=x)


class InvertedResBlock(AttnBottleneckBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_filters = self.filters
        self.exp_filters = self.filters * self.expansion
    
    def get_net(self, input_shape):
        inp = layers.Input(shape=input_shape[1:])

        x = ConvNorm(self.exp_filters, kernel_size=1, activation=self.activation, norm=self.norm)(inp)
        x = ConvNorm(self.exp_filters, kernel_size=3, strides=self.strides, activation=self.activation, norm=self.norm,
                     groups=self.exp_filters)(x)
        if self.squeeze_attn:
            x = SqueezeExcite()(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = ConvNorm(self.out_filters, kernel_size=1, activation=self.activation, norm=self.norm)(x)

        return tf.keras.Model(inputs=inp, outputs=x)

    def get_shortcut(self, input_shape):
        in_filters = input_shape[1]
        # operations for shortcut
        shortcut = tf.keras.Sequential()

        if self.strides>1:
            shortcut.add(layers.AveragePooling3D(self.strides, data_format="channels_first"))
        if in_filters != self.out_filters:
            shortcut.add(ConvNorm(self.out_filters, kernel_size=1, activation=self.activation, norm=self.norm))
        return shortcut

    def call(self, inp, training=False):
        x = inp

        identity = self.shortcut(x, training=training)

        x = self.net(x, training=training)
        x = x + identity
        return x