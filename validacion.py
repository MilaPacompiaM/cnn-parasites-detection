import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle



datos = open('/home/citesoft_barcos/Escritorio/tesis/model/datos_test.dat', 'rb')
datos_test = pickle.load(datos)
datos.close()

datos2 = open('/home/citesoft_barcos/Escritorio/tesis/model/label_test.dat', 'rb')
label_test = pickle.load(datos2)
datos2.close()


'''
new_model = tf.keras.models.load_model('/home/citesoft_barcos/Escritorio/tesis/modelos/modelo_91_25.h5')
print(new_model.history)

# Evaluate the restored model
loss, acc = new_model.evaluate(datos_test, label_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(datos_test).shape)

loss, accuracy  = new_model.evaluate(datos_test,label_test)

print( "Costo (Perdida) : " ,loss)
print( "Prediccion: " ,str(accuracy)+ "%")
#from keras.utils.vis_utils import plot_model  
#plot_model(new_model, show_shapes=True, show_layer_names=True)'''

model=tf.keras.models.load_model('/home/citesoft_barcos/Escritorio/tesis/model')
#model=tf.keras.models.load_model('/home/citesoft_barcos/Escritorio/tesis/model')
model.summary()
# Evaluate the restored model
#loss, acc = model.evaluate(datos_test, label_test, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#print(model.predict(datos_test).shape)
print("#####################################################")
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

history = np.load('/home/citesoft_barcos/Escritorio/tesis/model/history.npy').item()
#history = np.load('/home/citesoft_barcos/Escritorio/tesis/model/history.npy').item()
#print(history["loss"])
#Ver la funcion de perdida
'''
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(history["loss"])'''

loss, accuracy  = model.evaluate(datos_test,label_test)

print( "Costo (Perdida) : " ,(loss))
print( "Prediccion: " ,str(accuracy)+ "%")
'''
print('################################################################')
print('Accuracy', str(history['accuracy']))
print('Loss', str(history['loss']))
print('Val_Accuracy', str(history['val_accuracy']))
print('Val_Loss', str(history['val_loss']))
print('#################################################################')

plt.subplot(1,2,1)
plt.title('Perdida vs Iteracion')
plt.xlabel("Iteracion")
plt.ylabel("Perdida")
plt.plot(history["loss"])
#plt.plot(history["val_loss"])'''

plt.title('Precision y Perdida vs Iteracion')
plt.xlabel("Iteracion")
plt.ylabel("Perdida")
plt.plot(history["loss"])
plt.plot(history["accuracy"])
plt.plot(history["val_loss"])
plt.plot(history["val_accuracy"])
plt.legend(['loss','accuracy','val_loss','val_accuracy'])
print("Acc: " , history["accuracy"] )
print("val Acc : " , history["val_accuracy"])

plt.show()

