

vae_model.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = vae_model.fit(x_train, x_train, 
          epochs=200, 
          batch_size=32,
          validation_data=(x_test, x_test),
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

reconstructions = vae_model.predict(x_train)
train_loss = tf.keras.losses.mae(reconstructions, x_train)


from google.colab.patches import cv2_imshow

idx = 8

cv2_imshow(reconstructions[idx]*255)
cv2_imshow(x_train[idx]*255)
