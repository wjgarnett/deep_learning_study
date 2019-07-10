# coding: utf-8

from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from .shakespeare_utils import *


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
generate_output()
