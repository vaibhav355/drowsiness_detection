import os
import cv2
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to the GPU device index you want to use

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU devices available. Make sure TensorFlow is configured to use GPUs.")
    exit()

def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if (filename.endswith(".jpg") or filename.endswith(".png")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize the image to the target size (adjust as needed)
                image = cv2.resize(image, (224, 224))
                images.append(image)
                labels.append(label)
    return images, labels

dataset_dir = sys.argv[1]

open_folder = os.path.join(dataset_dir, "Open_Eyes")
closed_folder = os.path.join(dataset_dir, "Closed_Eyes")

if not os.path.exists(open_folder) or not os.path.exists(closed_folder):
    print("Error: 'Open' or 'Closed' folders not found.")
    exit()

open_eye_data, open_eye_labels = load_and_preprocess_images(open_folder, label=1)
closed_eye_data, closed_eye_labels = load_and_preprocess_images(closed_folder, label=0)

# Combine data and labels
data = np.vstack((open_eye_data, closed_eye_data))
labels = np.concatenate((open_eye_labels, closed_eye_labels))

# Print debug information
print("Number of samples in 'Open' class:", len(open_eye_data))
print("Number of samples in 'Closed' class:", len(closed_eye_data))
print("Total number of samples:", len(data))

# Split the data into training and testing sets
if len(data) >= 2:
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
else:
    print("Error: Insufficient data for splitting. Ensure you have at least 2 samples.")
    # Handle the error or adjust the dataset accordingly
    exit()

if len(train_data) >= 1:
    train_data = preprocess_input(train_data)
    test_data = preprocess_input(test_data)
else:
    print("Error: Insufficient data for preprocessing. Ensure you have at least 1 sample.")
    # Handle the error or adjust the dataset accordingly
    exit()
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential()
model.add(vgg_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 5
batch_size = 64
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2,validation_data=(test_data, test_labels),callbacks=[early_stopping])

# Evaluate the model on the testing data
test_predictions = model.predict(test_data)
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Calculate metrics
test_accuracy = accuracy_score(test_labels, test_predictions_binary)
test_precision = precision_score(test_labels, test_predictions_binary)
test_f1 = f1_score(test_labels, test_predictions_binary)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
model.save('Drowsiness_model.h5')

# You can later load the model using
loaded_model = load_model('Drowsiness_model.h5')
