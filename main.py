from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np 
from keras.utils import to_categorical
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator

train_dir = Path(f'./training/')
test_dir = Path(f'./validation/')

train_images = train_dir.glob('*.png')
test_images = test_dir.glob('*.png')
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
label_df = pd.read_csv(f"./monkey_labels.txt", names=cols, skiprows=1)
labels = label_df['Common Name']

height = 150
width = 150
batch_size = 32
seed = 100

#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=seed,
    shuffle=True,   
    class_mode='categorical')

test_datagen = ImageDataGenerator(
    rescale=1. / 255
    )
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=seed,
    shuffle=False,
    class_mode='categorical')

train_num = train_generator.samples
validation_num = validation_generator.samples 

from keras.models import Sequential
from keras.layers import Conv2D, Activation, GlobalAvgPool2D, MaxPooling2D, Flatten, Dense, Dropout , BatchNormalization
import os
from tensorflow import keras
neuralnet = Sequential()
neuralnet.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), strides=2))
neuralnet.add(Activation('relu'))
neuralnet.add(MaxPooling2D(pool_size=(2, 2)))
neuralnet.add(BatchNormalization())

neuralnet.add(Conv2D(32, (3, 3)))
neuralnet.add(Activation('relu'))
neuralnet.add(MaxPooling2D(pool_size=(2, 2)))
neuralnet.add(BatchNormalization())

neuralnet.add(Conv2D(64, (3, 3)))
neuralnet.add(Activation('relu'))
neuralnet.add(MaxPooling2D(pool_size=(2, 2)))
neuralnet.add(Conv2D(512, (1, 1), strides=2))
neuralnet.add(Activation('relu'))

neuralnet.add(Dropout(0.5))
neuralnet.add(Conv2D(10, (1, 1)))
neuralnet.add(GlobalAvgPool2D())
# neuralnet.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
neuralnet.add(Activation('softmax'))
# neuralnet.add(Dense(1, activation='softmax'))

neuralnet.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

checkpoint_path = "neuralnet_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

neuralnet.fit_generator(train_generator,
                        steps_per_epoch=train_num // batch_size,
                        epochs=30,
                        validation_data=train_generator,
                        validation_steps=validation_num // batch_size,
                        verbose=1,
                        callbacks=[cp_callback]
                        )


# os.listdir(checkpoint_dir)

# Save the entire model
neuralnet.save('neuralnet_model.h5')

# Load the model
loaded_model = keras.models.load_model('neuralnet_model.h5')


