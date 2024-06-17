from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np 
from keras.utils import to_categorical
import pandas as pd 
from tensorflow.image import rgb_to_grayscale, grayscale_to_rgb
from keras.preprocessing.image import ImageDataGenerator

train_dir = Path(f'./training/')
test_dir = Path(f'./validation/')

train_images = train_dir.glob('*.png')
test_images = test_dir.glob('*.png')
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
label_df = pd.read_csv(f"./monkey_labels.txt", names=cols, skiprows=1)
labels = label_df['Common Name']

def to_grayscale_then_rgb(image):
    image = rgb_to_grayscale(image)
    image = grayscale_to_rgb(image)
    return image

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
train_datagen_gray = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=to_grayscale_then_rgb,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=seed,
    shuffle=True,   
    class_mode='categorical')

train_generator_gray = train_datagen_gray.flow_from_directory(
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

def neural_model_1():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

def neural_model_2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['acc'])
    return model

checkpoint_path = f"net_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback1 = keras.callbacks.ModelCheckpoint(filepath='./color_model1.h5',
                                              save_weights_only=True,
                                              monitor='val_acc',
                                              mode='max',
                                              save_best_only=True,
                                              save_freq='epoch',
                                              verbose=1)
cp_callback2 = keras.callbacks.ModelCheckpoint(filepath='./color_model2.h5',
                                              save_weights_only=True,
                                              monitor='val_acc',
                                              mode='max',
                                              save_best_only=True,
                                              save_freq='epoch',
                                              verbose=1)
cp_callback3 = keras.callbacks.ModelCheckpoint(filepath='./gray_model1.h5',
                                              save_weights_only=True,
                                              monitor='val_acc',
                                              mode='max',
                                              save_best_only=True,
                                              save_freq='epoch',
                                              verbose=1)
cp_callback4 = keras.callbacks.ModelCheckpoint(filepath='./gray_model2.h5',
                                              save_weights_only=True,
                                              monitor='val_acc',
                                              mode='max',
                                              save_best_only=True,
                                              save_freq='epoch',
                                              verbose=1)


epochs = 100


colormodel1 = neural_model_1()

colormodel1.fit_generator(train_generator,
                        steps_per_epoch=train_num // batch_size,
                        epochs=epochs,
                        validation_data=train_generator,
                        validation_steps=validation_num // batch_size,
                        verbose=1,
                        callbacks=[cp_callback1]
                        )


# colormodel2 = neural_model_2()

# colormodel2.fit_generator(train_generator,
#                         steps_per_epoch=train_num // batch_size,
#                         epochs=epochs,
#                         validation_data=train_generator,
#                         validation_steps=validation_num // batch_size,
#                         verbose=1,
#                         callbacks=[cp_callback2]
#                         )

# graymodel1 = neural_model_1()
# graymodel1.fit_generator(train_generator_gray,
#                         steps_per_epoch=train_num // batch_size,
#                         epochs=epochs,
#                         validation_data=train_generator_gray,
#                         validation_steps=validation_num // batch_size,
#                         verbose=1,
#                         callbacks=[cp_callback3]
#                         )

# graymodel2 = neural_model_2()
# graymodel2.fit_generator(train_generator_gray,
#                         steps_per_epoch=train_num // batch_size,
#                         epochs=epochs,
#                         validation_data=train_generator_gray,
#                         validation_steps=validation_num // batch_size,
#                         verbose=1,
#                         callbacks=[cp_callback4]
#                         )

# os.listdir(checkpoint_dir)

# Save the entire model

#Load the model
# loaded_model = keras.models.load_model('neuralnet_model.h5')




def visualized_history(model, name):
    print(model.history.history.keys())
    acc = model.history.history['acc']
    val_acc = model.history.history['val_acc']
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.tight_layout()
    # plt.show()
    

    
visualized_history(colormodel1, 'color_model1')
# visualized_history(colormodel2, 'color_model2')
# visualized_history(graymodel1, 'gray_model1')
# visualized_history(graymodel2, 'gray_model2')


colormodel1.save('color_model1_2.h5')
# colormodel2.save('color_model2.h5')
# graymodel1.save('gray_model1.h5')
# graymodel2.save('gray_model2.h5')


colormodel1.evaluate_generator(validation_generator, validation_num // batch_size, verbose=2)
# colormodel2.evaluate_generator(validation_generator, validation_num // batch_size, verbose=2)
# graymodel1.evaluate_generator(validation_generator, validation_num // batch_size, verbose=2)
# graymodel2.evaluate_generator(validation_generator, validation_num // batch_size, verbose=2)

