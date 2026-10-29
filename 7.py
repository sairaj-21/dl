import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
dataset_path = pathlib.Path(data_dir) / "flower_photos"
print("Dataset Path:", dataset_path)

img_height, img_width, batch_size = 224, 224, 32

train_ds = image_dataset_from_directory(dataset_path, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size, label_mode="int")
val_ds = image_dataset_from_directory(dataset_path, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size, label_mode="int")

class_names = train_ds.class_names
print("\nClasses:", class_names)
print("\nNumber of Classes:", len(class_names))

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_dataset(image, label):
    return preprocess_input(tf.cast(image, tf.float32)), label

train_ds = train_ds.map(preprocess_dataset, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess_dataset, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
print("\nResNet50 loaded successfully.")

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(class_names), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\nModel Summary:")
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=5)

loss, accuracy = model.evaluate(val_ds)
print("\n===================================")
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy * 100, "%")
print("===================================")

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

from google.colab import files

print("\nUpload a Flower Image")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis("off")
plt.show()

img_array = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
img_array = preprocess_input(img_array)

prediction = model.predict(img_array)
predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("\n===================================")
print("Predicted Flower:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))
print("===================================")

print("\nPrediction Probabilities:")
for class_name, probability in zip(class_names, prediction[0]):
    print("{}: {:.2f}%".format(class_name, probability * 100))

model.save("flower_resnet50_model.keras")
print("\nModel saved successfully!")
print("File: flower_resnet50_model.keras")

files.download("flower_resnet50_model.keras")