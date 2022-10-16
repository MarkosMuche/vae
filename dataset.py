
from tensorflow import keras

data_dir = '/content/drive/MyDrive/meter_classification/data_2/meter'
data = keras.utils.image_dataset_from_directory(data_dir, labels = None, batch_size = 1000,  image_size=(224, 224),)
import math
x_train = data.__iter__().__next__().numpy()
data_length = x_train.shape[0]

x_test = x_train[math.floor(0.9*data_length):]

x_train = x_train[0:math.floor(0.9*data_length)]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

