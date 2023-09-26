# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:08:57 2023

@author: doguk
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
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
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(train_ds.class_names), activation='softmax')
])

# Compile the model
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer= optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
print(f"Precision: {precision_score(val_labels, predictions, average='weighted', zero_division=1)}")
print(f"Recall: {recall_score(val_labels, predictions, average='weighted')}")
print(f"F1 Score: {f1_score(val_labels, predictions, average='weighted')}")

# Plot Confusion Matrix using matshow instead of sns.heatmap
plt.figure(figsize=(10, 10))
cm = confusion_matrix(val_labels, predictions)
plt.matshow(cm, fignum=1, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(np.arange(len(train_ds.class_names)), train_ds.class_names, rotation=90)
plt.yticks(np.arange(len(train_ds.class_names)), train_ds.class_names)
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i][j], ha='center', va='center')
plt.show()

# Extract 4 random images and labels from the validation dataset
images, labels = [], []

for img, label in val_ds.unbatch().shuffle(buffer_size=100).take(4):
    images.append(img)
    labels.append(label)

# Visualize the images
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(f'Actual: {train_ds.class_names[labels[i]]}')
    plt.axis('off')
plt.show()