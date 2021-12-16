import json
import os
from glob import glob
import numpy as np
from typing import List
from AI import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys
import math
import argparse

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Flatten, Input, concatenate, Dense, Activation, Dropout, BatchNormalization,  MaxPooling2D, AveragePooling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='Test data folder mounting point')
parser.add_argument('--epochs', type=str, dest='epochs', help='Amount of epochs to train')
parser.add_argument('--batch_size', type=str, dest='batch_size', help='Batch size')
parser.add_argument('--model_name', type=str, dest='model_name', help='Model name')
args = parser.parse_args()

data_folder = args.data_folder

sys.path.insert(0, "./ai")

components_in_pack = ['Object 1', 'Object 3'] # The components you want to train

all_components = []
all_labels = []
all_sizes = []
for comp in components_in_pack:
    for img_uri in glob(os.path.join(data_folder, comp) + "/*.png"):
        try:
            size = img_uri.split(".png")[-2]
            with open(size + '--size.json', "r") as f:
                all_sizes.append(json.load(f))
                img = imread(img_uri)[:,:,:3] / 255
                all_components.append(img)
                all_labels.append(comp)
        except FileNotFoundError:
            pass

all_labels_np = np.array(all_labels)
all_components_np = np.array(all_components)
all_sizes_np = np.array(all_sizes)


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(all_labels_np)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


training_indices = []
test_indices = []
for obj in range(0, onehot_encoded.shape[1]):
    obj_indices = np.where(onehot_encoded[:,obj] == 1)[0]
    np.random.shuffle(obj_indices)
    training_samples = math.floor(0.7 * len(obj_indices))
    training_indices.extend(obj_indices[:training_samples])
    test_indices.extend(obj_indices[training_samples:])


X_train_conv = all_components_np[training_indices]
X_train_values = all_sizes_np[training_indices]
X_test_conv = all_components_np[test_indices]
X_test_values = all_sizes_np[test_indices]

y_train = onehot_encoded[training_indices]
y_test = onehot_encoded[test_indices]


# CNN - VGG
X_conv = Input(shape=(64, 64, 3))

vgg_model = VGG19(include_top=False, weights='imagenet')(X_conv)    # Add all the layers of the VGG19 model
# vgg_model.trainable = False
# vgg_model[-1].trainable = True
## Eventueel naar voor terugschuiven om op false te zetten

x_1 = Flatten(name='flatten')(vgg_model)
x_1 = Dense(512, activation='relu', name='fully-connected-1')(x_1)
x_1 = Dense(128, activation='relu', name='fully-connected-2')(x_1)
x_1 = Dense(16, activation='relu', name='fully-connected-3')(x_1)

# x_1 = Dense(6, activation='relu', name='fully-connected-TEST')(x_1)
# final_model = Model(inputs=X_conv, outputs=x_1)


## Output
x_1 = Dense(len(components_in_pack), activation='softmax', name='combined-fully-connected-2')(x_1)

final_model = Model(inputs=X_conv, outputs=x_1)

opt = tensorflow.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0, clipvalue=0.6)

final_model.compile(optimizer=opt, loss='categorical_crossentropy', 
                   metrics=['accuracy'])




train_generator = ImageDataGenerator(rotation_range=270)

batch_size = int(args.batch_size)
epochs = int(args.epochs)

time_to_repeat_generator = 5
vgg_generator = train_generator.flow([np.repeat(X_train_conv, time_to_repeat_generator, 0), np.repeat(X_train_values, time_to_repeat_generator, 0)], np.repeat(y_train, time_to_repeat_generator, 0), batch_size = batch_size)

early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1)

augmented_history = final_model.fit(vgg_generator,
                                    validation_data=([X_test_conv, X_test_values], y_test),
                                    epochs = epochs,
                                    steps_per_epoch = 8, # x * batch_size == amount of data in one epoch
                                    verbose = 1,
                                    shuffle=True,
                                    workers=1,
                                    callbacks=[early_stopping_callback, reduce_lr]
                                    )

final_model.save(f"outputs/{str(args.model_name)}")