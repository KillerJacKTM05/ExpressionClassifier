# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:08:57 2023

@author: doguk
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.experimental import preprocessing

#Allow gpu options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# Take epoch and batch size as console input
epochs = int(input("Enter the number of epochs: "))
batch_size = int(input("Enter the batch size: "))

# Load the data
data_dir = "./RAF-DB"  # Change this to the path of your RafD dataset
img_size = (100, 100)

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

# Add Data Augmentation
data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.1)
])

# Create the model
model = Sequential([
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_ds.class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model at the checkpoint where validation accuracy is maximum
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint])

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model and print the metrics
val_images, val_labels = next(iter(val_ds))
predictions = np.argmax(model.predict(val_images), axis=1)

print(f"Accuracy: {accuracy_score(val_labels, predictions)}")
print(f"Precision: {precision_score(val_labels, predictions, average='weighted')}")
print(f"Recall: {recall_score(val_labels, predictions, average='weighted')}")
print(f"F1 Score: {f1_score(val_labels, predictions, average='weighted')}")

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
cm = confusion_matrix(val_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_ds.class_names, yticklabels=train_ds.class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Plot a sample image with its predicted and actual label
plt.figure(figsize=(5, 5))
plt.imshow(val_images[0].numpy().astype('uint8'))
plt.title(f"Actual: {train_ds.class_names[val_labels[0]]}, Predicted: {train_ds.class_names[predictions[0]]}")
plt.axis('off')
plt.show()