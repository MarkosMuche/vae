

from numpy import imag
import tensorflow as tf
import cv2

model_path = '/media/mark/New Volume/projects/meter_classification/vae/saved_models/saved_model'
image_path = '/media/mark/New Volume/projects/meter_classification/vae/images/non_meter/0BW65ZVIB1.jpg'

vae_model = tf.keras.models.load_model(model_path)

image = cv2.imread(image_path)
image.resize((224,224,3), refcheck=False)
image = tf.expand_dims(image, axis=0)
cv2.imwrite('generated_images/original_image.jpg', image.numpy())

image = image/255
print(image.shape)


prediction = vae_model.predict(image)

# cv2.imshow('prediction', prediction[0]*255)
cv2.imwrite('generated_images/my_image.jpg', prediction[0]*255)