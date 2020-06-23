from keras import layers , models , optimizers, losses , metrics,initializers
from keras.utils  import  plot_model

import  numpy as np
from random import  shuffle

# 生成 LR 图。
feature_len = 1000 # len(val_x[0])

input = layers.Input(shape=(feature_len,),name="input")
output  = layers.Dense(1,
                       kernel_initializer=initializers.truncated_normal(),
                       activation='sigmoid')(input)

model = models.Model([input], output)

model.compile(
              optimizer=optimizers.Adagrad(lr=0.001),
              # optimizer = optimizers.sgd(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
model.summary()


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) # plot my model
# model.fit({"input":train_x}, train_y,
#           verbose=1, epochs=100, batch_size=256,
#           validation_data = ( {"input":val_x} , val_y ))

