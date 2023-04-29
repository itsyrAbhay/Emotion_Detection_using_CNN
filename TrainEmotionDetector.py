# import required packages for training our emotion detection model
# this keras library in the backend uses the tensorflow

import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# this image generators I will be using for preprocessing all the images
# Initialize image data generator with rescaling
# for training purpose
train_data_gen = ImageDataGenerator(rescale=1. / 255)
# for testing/validation purpose
validation_data_gen = ImageDataGenerator(rescale=1. / 255)
# flow_from_directory is a keras tool used to collect data and preprocess it
# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    )

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    )
# we used colormode as grayscale because if we used rgb it will take a lot more time
# training and testing/validation data is now preprocessed

# creating convolutional neural network(CNN)
# create model structure
# importing sequential model from keras and inside it adding different layers
# adding multiple convolutional layers
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# to avoid the overfitting we added a dropout with 0.25
# again adding these layers because cnn can contain many number of these layers

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# added flatten layer to flatten all the values

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# this layer helps in changing the dimensionality and define the relationship between the values of the data
# this layer is used in the final stages of nn
cv2.ocl.setUseOpenCL(False)
# compiling the convolutional layer with loss function, optimizer and metrics
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# training the cnn layer with our preprocessed data

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=45,
    validation_data=validation_generator,
    validation_steps=7178 // 64)
# steps per epoch is a total number of images divided by 64
# setting the epochs to 30 because of system capabilities, with better system you can go for higher
# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')
