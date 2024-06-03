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
batch_size = 64
seed = 100
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1. / 255)
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
from keras.layers import Conv2D, Activation, GlobalAvgPool2D

neuralnet = Sequential()
neuralnet.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), strides=2))
neuralnet.add(Activation('relu'))
neuralnet.add(Conv2D(10, (1, 1)))
neuralnet.add(GlobalAvgPool2D())
neuralnet.add(Activation('softmax'))

neuralnet.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

neuralnet.fit_generator(train_generator,
                        steps_per_epoch= train_num // batch_size,
                        epochs=10,
                        validation_data=train_generator,
                        validation_steps= validation_num // batch_size,
                        verbose = 1
                        )