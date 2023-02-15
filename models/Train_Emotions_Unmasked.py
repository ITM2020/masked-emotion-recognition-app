# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    rotation_range=15,
                                    zoom_range=0.2,
                                    shear_range=0.2,
                                    brightness_range=[0.2,1.8])
validation_data_gen = ImageDataGenerator(rescale=1./255,
                                         horizontal_flip=True,
                                         rotation_range=15,
                                         zoom_range=0.2,
                                         shear_range=0.2,
                                         brightness_range=[0.2,1.8])

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'Datasets/FER2013/train',
        target_size=(48, 48),
        batch_size=128,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'Datasets/FER2013/test',
        target_size=(48, 48),
        batch_size=128,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
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
emotion_model.add(Dense(4, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=19346 // 128,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=4796 // 128)

# plot model train & validation accuracy vs. epochs
plt.figure(figsize=[8,6])
plt.plot(emotion_model_info.history['accuracy'],'r',linewidth=3.0)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Unmasked Emotion Recognition Model: Accuracy', fontsize=16)
plt.show()

# plot training and validation loss vs. epochs
plt.figure(figsize=[8,6])
plt.plot(emotion_model_info.history['loss'], 'r',linewidth=3.0)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Unmasked Emotion Recognition Model: Loss', fontsize=16)
plt.show()

# save model structure in jason file
model_json = emotion_model.to_json()
with open("EmotionUnmaskedModel.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('EmotionUnmaskedModel.h5')