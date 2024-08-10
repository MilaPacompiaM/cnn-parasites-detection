import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from PIL import Image

#im = Image.open('imagen.png')
#im.save('imagen2.jpg', quality=95)

#kernel = np.ones((6,6), np.float32)/36
#kernel = np.ones((5,5), np.float32)/25

carpeta="/home/citesoft_barcos/Escritorio/tesis/nuevas_pruebas"
names=os.listdir(carpeta)

cont=1

for i in names:

    imagen_direccion="/home/citesoft_barcos/Escritorio/tesis/nuevas_pruebas/"+str(i)

    img =cv2.imread(imagen_direccion)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    copia=img

    #Filtros
    #img_filter = cv2.filter2D(img, -1, kernel)
    #blur = cv2.blur(img, (15,15))
    #gblur = cv2.GaussianBlur(img, (5,5), 0)
    #median = cv2.medianBlur(img, 5) # best for image noises
    #bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75) # best for improve quality

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #img =cv2.resize(img, (500,500))
    #img_filter =cv2.resize(img_filter, (500,500))
    #blur =cv2.resize(blur, (500,500))
    #gblur =cv2.resize(gblur, (500,500))
    #median =cv2.resize(median, (500,500))
    #bilateralFilter =cv2.resize(bilateralFilter, (500,500))
    img_gray =cv2.resize(img_gray, (500,500),interpolation=cv2.INTER_LANCZOS4)

    copia=cv2.resize(copia,(500,500),interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('copia',copia)

    #proceso de reconocimiento
    bilateralFilter=cv2.bilateralFilter(img_gray, 9, 75, 75)
    cv2.imshow('bilateralFilter',bilateralFilter)
    #bilateralFilter = cv2.bilateralFilter(img_gray, 9, 75, 75)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(bilateralFilter,kernel,iterations =3)
    cv2.imshow('erosion',erosion)
    dilation = cv2.dilate(erosion,kernel,iterations =2)
    cv2.imshow('dilation',dilation)
    _, frame5 = cv2.threshold(dilation,1,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('frame5',frame5)
    kernel2 = np.ones((20, 20), np.uint8)
    #kernel3 = np.ones((10, 10), np.uint8)
    #closing = cv2.morphologyEx(frame5, cv2.MORPH_HITMISS, kernel2)
    closing = cv2.morphologyEx(frame5, cv2.MORPH_DILATE, kernel2)
    cv2.imshow('MORPH_DILATE',closing)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
    cv2.imshow('MORPH_CLOSE',closing)

    #cv2.imshow('tozero_inv', frame5)
    #closing = cv2.morphologyEx(frame5, cv2.MORPH_CLOSE, kernel)
    canny=cv2.Canny(closing,50,180,None,3)
    cv2.imshow('canny',canny)
    

    (c,j)=cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #copia=cv2.drawContours(copia,c,-1,(255,0,0),2)

    #cv2.imshow("imagen procesada", dst)


    for i in c:
        m=cv2.moments(i)
        print("variable " +str(i) + " :" +str(m['m00']))
        if m["m00"] != 0:
            cx=int(m['m10']/m['m00'])
            cy=int(m['m01']/m['m00'])
        else:
            # set values as what you need in the situation
            cX, cY = 0, 0

        

        #cv2.circle(copia,(cx,cy),3,(0,0,255),-1)

        #if m["m00"] > 300:

        #AREA
        (x,y,w,h)=cv2.boundingRect(i)
        #cv2.rectangle(copia,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),3)
        #print(x,y,w,h)
        correccionX=23
        correccionY=23
        if x-correccionX<0:
            correccionX=x
        if y-correccionY<0:
            correccionY=y
            
        recorte=copia[y-correccionY:y+h+23,x-correccionX:x+w+18]

            

        #recorte=copia[y:y+h,x:x+w]
        #print("########################################")
        #print ("Y : "+ str(y )+ "X : " +str(x))
        #print(recorte)
        recorte=cv2.resize(recorte, (500,500),interpolation=cv2.INTER_LANCZOS4)
        path = '/home/citesoft_barcos/Escritorio/tesis/nuevo_pruebas_cortes/neg'
        path2 = '/home/citesoft_barcos/Escritorio/tesis/nuevo_pruebas_cortes/pos'
        '''if m["m00"] >= 1000:
            cv2.imwrite(os.path.join(path2 , 'recorte'+str(cont)+'.png'), recorte)
        else :
            cv2.imwrite(os.path.join(path , 'recorte'+str(cont)+'.png'), recorte)'''
        #cv2.imwrite('recorte'+str(cont)+'.png',recorte)
        cv2.imshow('recorte',recorte)
        cv2.imshow('Lines',copia)
            
        if cv2.waitKey(0):
            cont+=1    



    #cv2.imshow('Original',img)
    #cv2.imshow('Filtro', img_filter)
    #cv2.imshow('blur', blur)
    #cv2.imshow('Escala de grises', img_gray)
    #cv2.imshow('Lines',copia)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


