from keras import layers , models , optimizers, losses , metrics,initializers
from keras.engine import Layer
import keras.backend as K
"""
@description：这里定义通用的层。
"""

class FM(Layer):
    def __init__(self, output_dim=1, latent=16, **kwargs): # 输出为 1 个节点。
        print("output_dim:" + str(output_dim))  # 20
        self.latent = latent  # 10
        self.output_dim = output_dim  # 20
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input_shape:" + str(input_shape))  # [b,30]
        self.b = self.add_weight(name='W0',
                                 shape=(self.output_dim,),
                                 initializer='zeros')
        print("b:" + str(self.b.shape))  # [20,]
        self.w = self.add_weight(name='W',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer=initializers.truncated_normal())
        print("w:" + str(self.w.shape))  # [30,20]
        self.v = self.add_weight(name='V',
                                 shape=(input_shape[1], self.latent),
                                 initializer=initializers.truncated_normal())
        print("v:" + str(self.v.shape))  # [30,10]
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print("inputs:" + str(inputs.shape))
        x = inputs  # [b,30]

        xw = K.dot(x, self.w)  # xw=[b,20]
        print("xw:" + str(xw.shape))

        x_square = K.square(x)

        xv = K.square(K.dot(x, self.v))  # [b,10]
        print("xv:" + str(xv.shape))

        p = 0.5 * K.sum(xv - K.dot(x_square, K.square(self.v)), 1)  # [b,]
        print("p:" + str(p.shape))

        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)  # [b,20]
        print("rp:" + str(rp.shape))

        f = xw + rp + self.b  # [b,20] = [b,20] + [b,20] + [20]
        print("f:" + str(f.shape))

        # output = K.reshape(f, (-1, self.output_dim))
        # print("output:"+str(output.shape))

        return f

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim

