#!/usr/bin/env python3
import sys
import os
import cv2
import time
import pickle
from numpy import *
from config import load_config

def main():
    json_file_path = sys.argv[1]
    config = load_config(json_file_path) 
    data = f'{config.CUTTINGS_POSITIVES}'
    data2 = f'{config.CUTTINGS_NEGATIVES}'
    output_directory = f'{config.DATA_AUGMENTED}'

    #Cambiamos el orde de manera aleatoria a los arreglos
    def mezclar_lista(lista_original):
        #Establecemos listas auxiliares 
        lista = lista_original[:]
        #Recorremos el arreglo para cambiar el orden 
        for i in range(len(lista)):
            #Creamos un numero aleatorio
            indice_aleatorio = random.randint(0, len(lista) - 1)
            #Cambiamos de posicion la imagen
            temporal = lista[i]
            lista[i] = lista[indice_aleatorio]
            lista[indice_aleatorio] = temporal
        return (lista)


    ###########################################################################################
    # CARGA DE DATOS

    #Definimos los arreglos donde se guardaran los datos
    data_train=[] # positivo
    data_train2=[] # negativo


    #Extraemos los datos de la carpeta ( Valores Negativos)
    for i in os.listdir(data2):
        #Se lee la imagen 
        img=cv2.imread(os.path.join(data2,i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img= cv2.resize(img, (300,300),interpolation=cv2.INTER_LANCZOS4)
        #Se establece el tipo de dato
        img = img.astype("float32")
        #Se normaliza para una mejor ejecucion de la CNN
        img /= 255
        img=cv2.bilateralFilter(img, 9, 75, 75)
        #Se añade al arreglo correspondiente
        data_train2.append(img)


    print ("Termine carga Negativos ")

    #Se repite el proceso pero para la otra fuente de datos (Positivos)
    for i in os.listdir(data):
        
        img=cv2.imread(os.path.join(data,i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img= cv2.resize(img, (300,300),interpolation=cv2.INTER_LANCZOS4)
        img = img.astype("float32")
        img /= 255
        img=cv2.bilateralFilter(img, 9, 75, 75)
        data_train.append(img)
        #print ( "data_train " + str(len(data_train))) pkl

    print ("Termine carga Positivos ")


    ##########################################################################################33
    #Se mezcla los arreglos obtenidos para su alateorizacion

    data_train = mezclar_lista(data_train)
    print ("Termine Mezcla positivos ")
    data_train2 = mezclar_lista(data_train2)
    print ("Termine Mezcla negativos ")

    ###################################################################################################
    #Se obtiene el 80% de datos positivos para separar en entrenamiento - validacion
    length_train = int(len(data_train)*0.8)

    print ( "Total positivos: " + str(len(data_train)))

    datos_entranamiento = data_train[:length_train]
    datos_test = data_train[length_train:]

    print ( "Total inicial entrenamiento: " + str(len(datos_entranamiento)))

    print ( "Total inicial validacion:" + str(len(datos_test)))


    #Se realiza el data augmentation para los datos de entrenamiento 
    datos_aux = []
    for i in datos_entranamiento:
        img = i

        flipVertical = cv2.flip(img, 0)
        flipHorizontal = cv2.flip(img, 1)
        flipBoth = cv2.flip(img, -1)
        datos_aux.append(img)
        datos_aux.append(flipVertical)
        datos_aux.append(flipHorizontal)
        datos_aux.append(flipBoth)

    print ("Termine Aumenter datos positivos ")

    #Se realiza el data augmentation para los datos de validacion 
    datos_aux_test = []
    for i in datos_test:
        img = i

        flipVertical = cv2.flip(img, 0)
        flipHorizontal = cv2.flip(img, 1)
        flipBoth = cv2.flip(img, -1)
        datos_aux_test.append(img)
        datos_aux_test.append(flipVertical)
        datos_aux_test.append(flipHorizontal)
        datos_aux_test.append(flipBoth)
    print ("Termine Aumenter datos positivos 2")


    print ( "Imagenes para entrenamiento positivos: " + str(len(datos_aux)))
    print ( "Imagenes para validacion positivos: " + str(len(datos_aux_test)))


    #Se guarda los arreglos

    archivo = open(f'{output_directory}/{config.DATA_AUGMENTED_FILE_POSITIVE}{config.DATA_EXTENSION}', 'wb')
    pickle.dump(datos_aux, archivo)
    archivo.close()

    archivo = open(f'{output_directory}/{config.DATA_AUGMENTED_FILE_TEST_POSITIVE}{config.DATA_EXTENSION}', 'wb')
    pickle.dump(datos_aux_test, archivo)
    archivo.close()

    print ("######################################################### ")
    time.sleep(20)
    del datos_aux
    del datos_aux_test
    del datos_entranamiento
    del datos_test

    #####################################################################
    #Se obtiene el 80% de datos negativos para separar en entrenamiento - validacion
    leng_train = int(len(data_train2)*0.8)

    print ( "Total Negativos: " + str(len(data_train2)))
    datos_entranamiento = data_train2[:leng_train]
    datos_test = data_train2[leng_train:]

    print ( "Total inicial entrenamiento: " + str(len(datos_entranamiento)))

    print ( "Total inicial validacion:" + str(len(datos_test)))


    #Se realiza el data augmentation para los datos de entrenamiento 
    datos_aux = []
    for i in datos_entranamiento:
        img = i

        flipVertical = cv2.flip(img, 0)
        flipHorizontal = cv2.flip(img, 1)
        flipBoth = cv2.flip(img, -1)
        datos_aux.append(img)
        datos_aux.append(flipVertical)
        datos_aux.append(flipHorizontal)
        datos_aux.append(flipBoth)
    print ("Termine Aumenter datos negativos ")

    #Se realiza el data augmentation para los datos de validacion 
    datos_aux_test = []
    for i in datos_test:
        img = i

        flipVertical = cv2.flip(img, 0)
        flipHorizontal = cv2.flip(img, 1)
        flipBoth = cv2.flip(img, -1)
        datos_aux_test.append(img)
        datos_aux_test.append(flipVertical)
        datos_aux_test.append(flipHorizontal)
        datos_aux_test.append(flipBoth)
    print ("Termine Aumenter datos negativos 2")

    #Se guarda los arreglos
    print ( "Imagenes para entrenamiento Negativos: " + str(len(datos_aux)))
    print ( "Imagenes para validacion Negativos: " + str(len(datos_aux_test)))

    archivo = open(f'{output_directory}/{config.DATA_AUGMENTED_FILE_NEGATIVE}{config.DATA_EXTENSION}', 'wb')
    pickle.dump(datos_aux, archivo)
    archivo.close()

    archivo = open(f'{output_directory}/{config.DATA_AUGMENTED_FILE_TEST_NEGATIVE}{config.DATA_EXTENSION}', 'wb')
    pickle.dump(datos_aux_test, archivo)
    archivo.close()

if __name__ == "__main__":
    print('------ 2. DATA AUGMENTATION START ------')
    main()
    print('------ DATA AUGMENTATION END ------')