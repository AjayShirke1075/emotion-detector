import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load Dataset
csv_file = os.path.join("..", "data", "FER-2013", "fer2013.csv")
data = pd.read_csv(csv_file)

# Parameters
num_classes = 7
img_size = 48

# Preprocess data
def preprocess_data():
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(pix, sep=' ').reshape(img_size, img_size, 1) for pix in pixels])
    images = images.astype('float32') / 255.0
    emotions = to_categorical(data['emotion'], num_classes)
    return train_test_split(images, emotions, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = preprocess_data()

# Build CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_val, y_val))

# Save the model
model.save("face_emotion_model.h5")
print("Model trained and saved as face_emotion_model.h5")
