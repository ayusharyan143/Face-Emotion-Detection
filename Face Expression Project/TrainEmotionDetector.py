# import required packages 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt



# initalizer image data generator with rescaling 

train_data_gen = ImageDataGenerator(rescale = 1./255 ) 
validation_data_gen = ImageDataGenerator(rescale = 1./255 ) 


# preprocess all test images

train_generator = train_data_gen.flow_from_directory(
    r"C:\Users\ayush\Desktop\Mini Project GEHU\Dataset\FER-2013 Dataset\train",
    target_size=(48, 48),
    batch_size=64, 
    color_mode='grayscale', 
    class_mode='categorical'
)


# preprocess all train images

validation_generator = validation_data_gen.flow_from_directory(
    r"C:\Users\ayush\Desktop\Mini Project GEHU\Dataset\FER-2013 Dataset\test",
    target_size=(48, 48),
    batch_size=64, 
    color_mode='grayscale' ,
    class_mode='categorical'
)



# Create the Convolutional Neural Network (CNN) model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate=0.0001), 
                      metrics=['accuracy'])


print(device_lib.list_local_devices())

# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64
)


# Save the training history
history = emotion_model_info.history





# Save model structure in JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights in .weights.h5 file
emotion_model.save_weights('emotion_model.weights.h5')  # Changed the filename


import matplotlib.pyplot as plt
import numpy as np

# Simulating training history for demonstration purposes
# In practice, replace this with your actual model training history
epochs = 50
emotion_model_info = {
    'accuracy': np.random.rand(epochs),        # Replace with your actual training accuracy
    'val_accuracy': np.random.rand(epochs),    # Replace with your actual validation accuracy
    'loss': np.random.rand(epochs),             # Replace with your actual training loss
    'val_loss': np.random.rand(epochs)          # Replace with your actual validation loss
}

# Extract training and validation metrics
train_accuracy = emotion_model_info['accuracy']
val_accuracy = emotion_model_info['val_accuracy']
train_loss = emotion_model_info['loss']
val_loss = emotion_model_info['val_loss']

# Print the values in a detailed format
print("Training and Validation Metrics:")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}:")
    print(f"  Training Accuracy: {train_accuracy[epoch]:.4f}")
    print(f"  Validation Accuracy: {val_accuracy[epoch]:.4f}")
    print(f"  Training Loss: {train_loss[epoch]:.4f}")
    print(f"  Validation Loss: {val_loss[epoch]:.4f}")
    print()  # Add a blank line for better readability


# Create subplots for accuracy and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot Accuracy
axs[0].plot(train_accuracy, label='Training Accuracy', color='blue', marker='o')
axs[0].plot(val_accuracy, label='Validation Accuracy', color='orange', marker='o')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].set_ylim(0, 1)  # Assuming accuracy is between 0 and 1
axs[0].legend(loc='lower right')
axs[0].grid()

# Plot Loss
axs[1].plot(train_loss, label='Training Loss', color='blue', marker='o')
axs[1].plot(val_loss, label='Validation Loss', color='orange', marker='o')
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper right')
axs[1].grid()

# Show the plots
plt.tight_layout()
plt.show()


