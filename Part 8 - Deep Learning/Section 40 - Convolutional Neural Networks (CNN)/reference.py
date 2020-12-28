# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:38:34 2019

@author: amrut
"""
# Convolutional Neural Network

# keras

# Part 1 --> Building the Convolution Nueral Network

# Importing the Keras libraries and packages

from keras.models import Sequential  # For initialising our CNN as initialisation can be done in two was that is sequential and graphical but since CNN is sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

 #Initialising the CNN
 
classifier = Sequential()
 
 # Step 1 -- Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu')) # as we are using the TEnserFlow packet and hence the arguements are accepted in the reverse order
 
# Step 2 --> MAx Pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second Convolution layer 

classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) #since keras knows the images now

classifier.add(MaxPooling2D(pool_size=(2,2)))
# Step 3 --> Flattening

classifier.add(Flatten())


#Step 4--> Full COnnections

classifier.add(Dense(output_dim=128, activation = 'relu' ))  #HIDDEN LAYER
classifier.add(Dense(output_dim=1, activation = 'sigmoid' )) #OUTPUT LAYER

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2--> Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                   'dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                                    'dataset/test_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

classifier.fit_generator(
                            training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)




















