# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:08:57 2023

@author: doguk
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

#Allow gpu options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# Take epoch and batch size as console input
epochs = int(input("Enter the number of epochs: "))
batch_size = int(input("Enter the batch size: "))

# Load the data
data_dir = "./RAF-DB"
img_size = (100, 100)
file_paths = []
labels = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):  # Assuming the images are in jpg format
            path = os.path.join(root, file)
            label = root.split("\\")[-1]  # Assuming the folder name is the label
            file_paths.append(path)
            labels.append(label)

# Perform stratified split
train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.25, stratify=labels, random_state=123)

# Create Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
)

val_ds = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_paths, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
)

# Create the model
num_classes = len(train_ds.class_indices)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model at the checkpoint where validation accuracy is maximum
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint])

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model and print the metrics
class_names = list(train_ds.class_indices.keys())
val_images, val_labels_one_hot = next(iter(val_ds))
val_labels = np.argmax(val_labels_one_hot, axis=1)
predictions = np.argmax(model.predict(val_images), axis=1)

print("Unique labels in val_labels:", np.unique(val_labels))
print("Class names: ", class_names)
print("Unique predictions: ", np.unique(predictions))

print(f"Accuracy: {accuracy_score(val_labels, predictions)}")
print(f"Precision: {precision_score(val_labels, predictions, average='weighted', zero_division=1)}")
print(f"Recall: {recall_score(val_labels, predictions, average='weighted')}")
print(f"F1 Score: {f1_score(val_labels, predictions, average='weighted')}")

# Plot Confusion Matrix using matshow
plt.figure(figsize=(10, 10))
cm = confusion_matrix(val_labels, predictions)
plt.matshow(cm, fignum=1, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(np.arange(num_classes), class_names, rotation=90)
plt.yticks(np.arange(num_classes), class_names)
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i][j], ha='center', va='center')
plt.show()

# Extract 4 random images and labels from the validation dataset
pred = []
# Fetch a batch of images and labels from validation dataset
images, labels = next(val_ds)
# Find the unique labels and their corresponding indices in the batch
unique_labels, indices = np.unique(labels, return_index=True)

# If there are at least 4 unique labels, select the first image of each unique label
if len(indices) >= 4:
    selected_images = images[indices[:4]]
    selected_labels = unique_labels[:4]
# If there are less than 4 unique labels, randomly select 4 images from the batch
else:
    indices = np.random.choice(len(images), 4, replace=False)
    selected_images = images[indices]
    selected_labels = labels[indices]
    
# Visualize the selected images
plt.figure(figsize=(10, 10))
for i in range(4):
    img = selected_images[i]
    label_one_hot = selected_labels[i]  # this is one-hot encoded
    label_index = np.argmax(label_one_hot)  # converting one-hot to index
    pred = model.predict(tf.expand_dims(img, 0))
    pred_label = np.argmax(pred, axis=-1)[0]
    
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.squeeze(img))
    plt.title(f'Actual: {class_names[label_index]} Predicted: {class_names[pred_label]}')
    plt.axis('off')

plt.show()