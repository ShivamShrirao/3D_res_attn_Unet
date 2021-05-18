import tensorflow as tf
from tensorflow.keras import layers


class AbsPosEmb(layers.Layer):
    def __init__(self, dhw, **kwargs):
        super().__init__(**kwargs)
        self.dhw = dhw

    def build(self, input_shape):
        dim = input_shape[-2]         # [B, N, dv/N, D*H*W]
        D,H,W = self.dhw
        scale = dim**-0.5
        self.emb_d = self.add_weight("emb_d", shape=(1, 1, dim, D, 1, 1), initializer="truncated_normal", trainable=True)
        self.emb_h = self.add_weight("emb_h", shape=(1, 1, dim, 1, H, 1), initializer="truncated_normal", trainable=True)
        self.emb_w = self.add_weight("emb_w", shape=(1, 1, dim, 1, 1, W), initializer="truncated_normal", trainable=True)
        self.restore_shape = layers.Reshape(target_shape=(1, dim, -1))

    def call(self, q):
        emb = self.emb_d + self.emb_h + self.emb_w      # [1, 1, dv/N, D, H, W]
        emb = self.restore_shape(emb)     # [1, 1, dv/N, D*H*W]
        # [B, N, dv/N, D*H*W] @ [1, 1, dv/N, D*H*W] ->  [B, N, D*H*W, D*H*W]
        return tf.matmul(q, emb, transpose_a=True)

    def get_config(self):
        l_config = {"dhw": self.dhw}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(l_config.items()))


class MHSA3D(layers.Layer):
    def __init__(self, dv=None, nheads=8, prev_kq=None, **kwargs):
        super().__init__(**kwargs)
        self.dv = dv
        self.scale = (dv//nheads)**-0.5
        self.nheads = nheads
        self.to_qkv = layers.Conv3D(filters=3*dv, kernel_size=1, strides=1, padding='same',
                                    data_format='channels_first')

    def build(self, input_shape):
        B, C, D, H, W = input_shape
        self.flatten = layers.Reshape(target_shape=(self.nheads, -1, D*H*W))
        self.pos_emb = AbsPosEmb((D,H,W))
        self.softmax = layers.Softmax(axis=-1)
        self.restore_shape = layers.Reshape(target_shape=(-1, D, H, W))

    def call(self, inp):
        q, k, v = tf.split(self.to_qkv(inp), [self.dv, self.dv, self.dv], axis=1)    # [B, dv, D, H, W]
        q, k, v = [self.flatten(x) for x in (q,k,v)]                    # [B, N, dv/N, D*H*W]
        q *= self.scale
        qk = tf.matmul(q, k, transpose_a=True)      # [B, N, dv/N, D*H*W] @ [B, N, dv/N, D*H*W] -> [B, N, D*H*W, D*H*W]
        qk += self.pos_emb(q)         # [B, N, D*H*W, D*H*W]

        attn = self.softmax(qk)         # [B, N, D*H*W, D*H*W]

        # [B, N, dv/N, D*H*W] @ [B, N, D*H*W, D*H*W] -> [B, N, dv/N, D*H*W]
        out = tf.matmul(v, attn, transpose_b=True)  # [B, N, dv/N, D*H*W]
        out = self.restore_shape(out)   # [B, dv, D, H, W]
        return out

    def get_config(self):
        l_config = {"dhw": self.dhw}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(l_config.items()))


class SqueezeExcite(layers.Layer):
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gpool = layers.GlobalAveragePooling3D(data_format='channels_first')

    def build(self, input_shape):
        self.input_channels = input_shape[-4]
        self.fc1 = layers.Dense(self.input_channels//self.ratio, activation='relu')
        self.fc2 = layers.Dense(self.input_channels, activation='hard_sigmoid')
        self.restore_shape = layers.Reshape(target_shape=(self.input_channels, 1, 1, 1))    # [B, C, D, H, W]

    def call(self, inp=False):
        x = self.gpool(inp)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.restore_shape(x)
        x = inp * x
        return x

    def get_config(self):
        l_config = {"ratio": self.ratio}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(l_config.items()))