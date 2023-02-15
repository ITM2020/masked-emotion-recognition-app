from matplotlib import pyplot as plt
from keras.applications.vgg19 import VGG19
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'Datasets/FaceMaskDataset/Train'
test_dir = 'Datasets/FaceMaskDataset/Test'
val_dir = 'Datasets/FaceMaskDataset/Validation'

train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in vgg19.layers:
    layer.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics =['accuracy'])


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_generator)//32,
                              epochs=20,
                              validation_data=val_generator,
                              validation_steps=len(val_generator)//32)

# plot model train & validation accuracy vs. epochs
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Mask Detection Model: Validation Accuracy', fontsize=16)
plt.show()

# plot training and validation loss vs. epochs
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'], 'r',linewidth=3.0)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Mask Detection Model: Validation Loss', fontsize=16)
plt.show()

# save model structure in jason file
model_json = model.to_json()
with open("MaskModel.json", "w") as json_file:
    json_file.write(model_json)

model.save('MaskModel.h5')