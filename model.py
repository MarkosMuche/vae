



import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential


my_model = keras.applications.vgg16.VGG16()


encoder = Sequential()

layers_to_freeze = 16
for i, layer in enumerate(my_model.layers[:-4]):
    if i<layers_to_freeze:
        layer.trainable = False
    else:
      layer.trainable = True
    encoder.add(layer)



decoder = Sequential()

decoder.add(keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same', input_shape=(7,7,512)))
decoder.add(keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
decoder.add(keras.layers.UpSampling2D((2,2)))
decoder.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
decoder.add(keras.layers.UpSampling2D((2,2)))
decoder.add(keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
decoder.add(keras.layers.UpSampling2D((2,2)))
decoder.add(keras.layers.Conv2D(16, (3,3), activation = 'relu', padding = 'same'))
decoder.add(keras.layers.UpSampling2D((2,2)))
decoder.add(keras.layers.Conv2D(3, (3,3), activation = 'relu', padding = 'same'))
decoder.add(keras.layers.UpSampling2D((2,2)))


input = encoder.input

output = decoder.output

vae_model = encoder

for layer in decoder.layers:

    vae_model.add(layer)

vae_model.summary()