import pickle
import random
import numpy as np
import time
import cv2
####################################################################3
#Carga de datos de entrenamiento
datos = open('datos/datos.dat', 'rb')
data_train = pickle.load(datos)
#data_train2 = np.asarray(lista)
datos.close()

datos2 = open('datos/datos2.dat', 'rb')
data_train2 = pickle.load(datos2)
#data_train = np.asarray(lista)
datos2.close()

#################################################################33
# Ajuste de datos

print( "Entrando a creacion de arrays")
#Se crea el arreglo de las etiquetas de 1 y 0
label_Y = np.ones(len(data_train))
label_Y2 = np.zeros(len(data_train2))

#Se combinan los dos arreglos de los datos y las etiquetas
datos = data_train+data_train2
label = np.append(label_Y,label_Y2)

#Eliminamos las varaibles que ya no se usaran
del data_train
del data_train2

del label_Y
del label_Y2

#####################################################################
#Mezcla de datos de entrenamiento

#Cambiamos el orde de manera aleatoria a los arreglos
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
  
print( "Entrando a mezcla")
(datos,etiquetas) = mezclar_lista(datos,label)
print( "Saliendo de  mezcla")

#Tamaño de los datos de entrenamiento
length_train = int(len(datos)*0.8)


#Datos de entrenamiento
print( "Creacion array")
datos_entranamiento = datos[:length_train]
label_entranamiento  = etiquetas[:length_train]

#Datos de pruebas
datos_test = datos[length_train:]
label_test  = etiquetas[length_train:]

del datos
del etiquetas

print ( "Imagenes para entrenamiento: " + str(len(datos_entranamiento)))

print ( "Imagenes para testeo: " + str(len(datos_test)))   


print( "Creacion np")
datos_entranamiento = np.array(datos)
label_entranamiento = np.array(etiquetas)

del datos
del etiquetas

#######################################################################
#Carga de datos de validacion

datos = open('datos/datos_test.dat', 'rb')
data_test = pickle.load(datos)
datos.close()

datos2 = open('datos/datos_test2.dat', 'rb')
data_test2 = pickle.load(datos2)
datos2.close()

print ("Termine de cargar datos de Validacion")
####################################################################
#Ajuste de datos

label_Y = np.ones(len(data_test))
label_Y2 = np.zeros(len(data_test2))

#Se combinan los dos arreglos de los datos y las etiquetas
datos = data_test+data_test2
label = np.append(label_Y,label_Y2)

del data_test
del data_test2

(datos,etiquetas) = mezclar_lista(datos,label)
print( "Termine de combinar datos de validacion  ")

time.sleep(20)



datos_test = np.array(datos)
label_test  = np.array(etiquetas)
print( "Termine de crear arrays de validacion  ")
  
print ( "Imagenes para entrenamiento: " + str(len(datos_entranamiento)))
print ( "Imagenes para validacion: " + str(len(datos_test)))


time.sleep(20)


##################################################################
archivo = open('datos/datos_entranamiento.dat', 'wb')
pickle.dump(datos_entranamiento, archivo)
archivo.close()
del datos_entranamiento
time.sleep(20)

archivo2 = open('datos/label_entranamiento.dat', 'wb')
pickle.dump(label_entranamiento, archivo2)
archivo2.close()
del label_entranamiento
time.sleep(20)

archivo3 = open('datos/datos_test.dat', 'wb')
pickle.dump(datos_test, archivo3)
archivo3.close()
del datos_test
time.sleep(20)

archivo4 = open('datos/label_test.dat', 'wb')
pickle.dump(label_test, archivo4)
archivo4.close()
del label_test
time.sleep(20)