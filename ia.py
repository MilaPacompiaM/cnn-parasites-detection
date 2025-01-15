#!/usr/bin/env python3
import numpy as np
import os
import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import tensorflow as tf
from enum import Enum

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
input_directory = f'{current_dir}/output-mezcla-array'
output_directory = f'{current_dir}/output-ia'

class FileName(Enum):
    # Input files
    DATA_TRAINING = 'datos_entrenamiento.dat'
    DATA_TRAINING_LABEL = 'label_entrenamiento.dat'
    DATA_TEST = 'datos_test.dat'
    DATA_TEST_LABEL = 'label_test.dat'

############################################################### 
# Carga de datos
datos = open(f'{input_directory}/{FileName.DATA_TRAINING.value}', 'rb')
datos_entranamiento = pickle.load(datos)
#data_train2 = np.asarray(lista)
datos.close()

datos2 = open(f'{input_directory}/{FileName.DATA_TRAINING_LABEL.value}', 'rb')
label_entranamiento = pickle.load(datos2)
#data_train = np.asarray(lista)
datos2.close()

print("Entrenamiento: " + str(len(datos_entranamiento)))
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

############################################################### 
# CNN
with tf.device('/GPU:0'):
    model = Sequential()

    #Convolution 
    model.add(Conv2D(32, (3,3), input_shape = (500,500,1),activation='relu')) 
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (3,3), activation='relu')) 
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    #Para el dropout


    #Flattening 
    model.add(Flatten())
    
    #Capas de la red
    model.add(Dense(60, input_shape=datos_entranamiento.shape[1:] , activation='relu'))
    model.add(Dense(32 ,activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    

    # Backpropagation
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenamiento
    history = model.fit(datos_entranamiento, label_entranamiento,steps_per_epoch=180,  epochs=13, validation_split=0.2)

    model.save(f'{output_directory}/{current_date}-model.keras')
    # model.save("model.h5")
    model.save_weights(f'{output_directory}/{current_date}-model.weights.h5')
    #with open('/trainHistoryDict', 'wb') as file_pi:
    #    pickle.dump(history.history, file_pi)

    np.save(f'{output_directory}/{current_date}-history.npy', history.history, allow_pickle=True) 