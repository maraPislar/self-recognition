#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:25:46 2022

@author: mara
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input

# Constants set for the model
HEIGHT = 255
WIDTH = 255
TRAIN_DIR = "self-recognition/face-rec/data/train"
VAL_DIR = "self-recognition/face-rec/data/val"
TEST_DIR = "self-recognition/face-rec/data/test"
BATCH_SIZE = 32
NUMBER_OF_CLASSES = 2
NUM_EPOCHS = 10
MODEL_NAME = "densenet169"


base_model = DenseNet169(weights = 'imagenet',
                      include_top = False,
                      input_shape = (HEIGHT, WIDTH, 3))


train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   rotation_range = 90,
                                   horizontal_flip = True,
                                   vertical_flip = False)

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                  rotation_range = 90,
                                  horizontal_flip = True,
                                  vertical_flip = False)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size = BATCH_SIZE)

val_generator = val_datagen.flow_from_directory(VAL_DIR,
                                                target_size = (HEIGHT, WIDTH),
                                                batch_size = BATCH_SIZE)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                  target_size= (HEIGHT, WIDTH),
                                                  batch_size = BATCH_SIZE)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
finetune_model = Model(inputs = base_model.input, outputs = x)

finetune_model.compile(optimizer=Adam(learning_rate = 0.01), 
                       loss="categorical_crossentropy", 
                       metrics=["acc"])

# Begin training the model
history = finetune_model.fit(train_generator, 
                             epochs = NUM_EPOCHS, 
                             shuffle = True, 
                             validation_data = val_generator)

# Save neural network structure
model_structure = finetune_model.to_json()
f = Path(MODEL_NAME + "_model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
finetune_model.save_weights(MODEL_NAME + "_model_weights.h5")