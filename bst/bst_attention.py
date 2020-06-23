
from keras.engine.topology import Layer
from keras import backend as K
from keras import layers

class Multy_Head_Attention(Layer): # 默认 8 头 ； 输入输出 shape 相同。

    def __init__(self, inner_dim = 64, head_num = 8, **kwargs):
        self.inner_dim = inner_dim # 64
        self.head_num = head_num   # 8
        super(Multy_Head_Attention, self).__init__(**kwargs)

    def build(self, input_shape): # [None,20,192]
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        #                [None,20,192]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3 * self.head_num, input_shape[2], self.inner_dim),
                                      initializer='uniform',
                                      trainable=True)
                                        # [3*8 , 192 , 64 ]
        self.head_kernel = self.add_weight(name='head_kernel',
                                      shape=(self.inner_dim * self.head_num , input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
                                        #[64*8 , 192]
        super(Multy_Head_Attention, self).build(input_shape)  # 一定要在最后调用

    def call(self, x): # [?,20,192]
        print("x-shape " + str(x.shape))
        # 012  345  678  9,10,11
        v_list = []
        for i in range(self.head_num):
            Q_idx = i * 3
            K_idx = i * 3 + 1
            V_idx = i * 3 + 2
            WQ = K.dot(x, self.kernel[Q_idx])#  [?,20,192] @ [192,64]
            WK = K.dot(x, self.kernel[K_idx])
            WV = K.dot(x, self.kernel[V_idx])
            print("WQ.shape", WQ.shape)     # [?,20,64]
            print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
            QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1])) # [?,20,192]@[?,192,20]
            QK = QK / (self.inner_dim ** 0.5)
            QK = K.softmax(QK)
            V = K.batch_dot(QK, WV)
            v_list.append(V)
        concated_v = layers.concatenate(v_list, axis=-1) # (? , 31 , inner_dim * head_num)
        V = K.dot(concated_v, self.head_kernel)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])
