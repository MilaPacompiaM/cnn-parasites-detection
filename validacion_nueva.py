
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import random
def mezclar_lista(lista_original,listaux):
    #Establecemos listas auxiliares 
    lista = lista_original[:]
    lista2 = listaux[:]
    #Recorremos el arreglo para cambiar el orden 
    for i in range(len(lista)):
        #Creamos un numero aleatorio
        indice_aleatorio = random.randint(0, len(lista) - 1)
        #Cambiamos de posicion la imagen
        temporal = lista[i]
        lista[i] = lista[indice_aleatorio]
        lista[indice_aleatorio] = temporal
        #Cambiamos de posicion su etiqueta
        temporal = lista2[i]
        lista2[i] = lista2[indice_aleatorio]
        lista2[indice_aleatorio] = temporal
    return (lista,lista2)


carpeta="/home/citesoft_barcos/Escritorio/tesis/nuevo_pruebas_cortes/neg"
names=os.listdir(carpeta)
carpeta2="/home/citesoft_barcos/Escritorio/tesis/nuevo_pruebas_cortes/pos"
names2=os.listdir(carpeta2)

cont=1
data = []
data2 = []
for i in names:
    img=cv2.imread(os.path.join(carpeta,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img= cv2.resize(img, (300,300),interpolation=cv2.INTER_LANCZOS4)
    #Se establece el tipo de dato
    img = img.astype("float32")
    #Se normaliza para una mejor ejecucion de la CNN
    img /= 255
    #img=cv2.bilateralFilter(img, 9, 75, 75)
    #Se añade al arreglo correspondiente
    flipVertical = cv2.flip(img, 0)
    flipHorizontal = cv2.flip(img, 1)
    flipBoth = cv2.flip(img, -1)
    
    data.append(img)
    data.append(flipVertical)
    data.append(flipHorizontal)
    data.append(flipBoth)
            


for i in names2:

    img=cv2.imread(os.path.join(carpeta2,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img= cv2.resize(img, (300,300),interpolation=cv2.INTER_LANCZOS4)
    #Se establece el tipo de dato
    img = img.astype("float32")
    #Se normaliza para una mejor ejecucion de la CNN
    img /= 255
    #img=cv2.bilateralFilter(img, 9, 75, 75)
    #Se añade al arreglo correspondiente
    flipVertical = cv2.flip(img, 0)
    flipHorizontal = cv2.flip(img, 1)
    flipBoth = cv2.flip(img, -1)
    data2.append(img)
    data2.append(flipVertical)
    data2.append(flipHorizontal)
    data2.append(flipBoth)

label_Y = np.ones(len(data2))
label_Y2 = np.zeros(len(data))

#Se combinan los dos arreglos de los datos y las etiquetas
datos = data2+data
label = np.append(label_Y,label_Y2)

(datos,etiquetas) = mezclar_lista(datos,label)
#shuffler = np.random.permutation(len(datos))
#arr_1_shuffled = datos[shuffler]
#arr_2_shuffled = label[shuffler]


model=tf.keras.models.load_model('/home/citesoft_barcos/Escritorio/tesis/modelos/model_bueno2')
print( "Entrando a mezcla")

# Evaluate the restored model
#loss, acc = model.evaluate(datos_test, label_test, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#print(model.predict(datos_test).shape)
print("#####################################################")
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#history = np.load('/home/citesoft_barcos/Escritorio/tesis/model/history.npy').item()
#print(history["loss"])

#Ver la funcion de perdida
'''
i = 0
for var in datos:
    cv2.imshow('datos',var)
    print ( "VALOR ES : " + str(etiquetas[i]))
    i+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
datos = np.array(datos)
etiquetas = np.array(label)

loss, accuracy  = model.evaluate(datos,etiquetas)
print("cantidad de datos : ", len(datos) )
print( "Costo (Perdida) : " ,(loss))
print( "Prediccion: " ,str(accuracy)+ "%")
