import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import skimage.exposure
from PIL import Image

carpeta="/home/citesoft_barcos/Escritorio/tesis/Data_IA/Ascaris/"
names=os.listdir(carpeta)

cont=1
aux = 0
for i in names:

    imagen_direccion="/home/citesoft_barcos/Escritorio/tesis/Data_IA/Ascaris/"+str(i)
    #imagen_direccion="/home/citesoft_barcos/Escritorio/tesis/Ascaris lumbricoides_0310.jpg"
   
    img =cv2.imread(imagen_direccion)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(500,500),interpolation=cv2.INTER_LANCZOS4)
    copia=img
    #########################################
    # QUITADO DE NEGRO
    hh, ww = img.shape[:2]

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #img_blured = cv2.blur(img_gray, (5, 5))
    ret, thres = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)

    # invert mask so shapes are white on black background
    thresh_inv = thres
    #cv2.imshow('thres', thres)
    # get the largest contour
    '''contours = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)'''

    # draw white contour on black background as mask
    mask = np.zeros((hh,ww), dtype=np.uint8)
    #cv2.drawContours(mask, [big_contour], 0, (255,255,255), cv2.FILLED)
    mask = thres
    # invert mask so shapes are white on black background
    mask_inv = 255 - mask

    '''average_color_row = np.average(img, axis=0)
    average_color = np.average(average_color_row, axis=0)
    print(average_color)'''

    # create new (blue) background
    #bckgnd = np.full_like(img, (average_color[0] - 20 ,average_color[1] - 20,average_color[2] - 20))
    bckgnd = np.full_like(img, (210 ,210,210))

    # apply mask to image
    image_masked = cv2.bitwise_and(img, img, mask=mask)

    # apply inverse mask to background
    bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)

    # add together
    result = cv2.add(image_masked, bckgnd_masked)
    #cv2.imshow('mask', mask)
    #cv2.imshow('img', img)
    #cv2.imshow('result', result)
    print("Nombre : " +str(i) )

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    ############################################################################
    # Preṕrocesamiento 

    #cv2.imshow("final", img)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result,(500,500),interpolation=cv2.INTER_LANCZOS4)
    #bilateralFilter=cv2.bilateralFilter(backtorgb, 9, 75, 75)
    bilateralFilter = cv2.bilateralFilter(result, 9, 75, 75)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(bilateralFilter,kernel,iterations =3)
    dilation = cv2.dilate(erosion,kernel,iterations =1)
    cv2.imshow('dilation',dilation)
    _, frame5 = cv2.threshold(dilation,1,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('bitsu',frame5)
    kernel2 = np.ones((20, 20), np.uint8)
    kernel3 = np.ones((5, 5), np.uint8)
    #closing = cv2.morphologyEx(frame5, cv2.MORPH_HITMISS, kernel2)
    closing = cv2.morphologyEx(frame5, cv2.MORPH_DILATE, kernel3)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
    #cv2.imshow('closing',closing)

    #cv2.imshow('tozero_inv', frame5)
    #closing = cv2.morphologyEx(frame5, cv2.MORPH_CLOSE, kernel)
    canny=cv2.Canny(closing,50,180,None,3)
    cv2.imshow('canny',canny)
    (c,j)=cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #copia=cv2.drawContours(copia,c,-1,(255,0,0),2)

    #cv2.imshow("imagen procesada", dst)

    #Codigo para Obtener los positivos / Negativos
    '''for i in c:
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
        path = '/home/citesoft_barcos/Escritorio/tesis/Data_IA_Procesada/Negativo'
        path2 = '/home/citesoft_barcos/Escritorio/tesis/Data_IA_Procesada/Positivo'
        #if m["m00"] >= 1000:
        #    cv2.imwrite(os.path.join(path2 , 'recorte'+str(cont)+'.png'), recorte)
        #else :
        #    cv2.imwrite(os.path.join(path , 'recorte'+str(cont)+'.png'), recorte)
        #cv2.imwrite('recorte'+str(cont)+'.png',recorte)
        cv2.imshow('recorte',recorte)
        #cv2.imshow('Lines',copia)
            
        if cv2.waitKey(0):
            cont+=1'''  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    aux +=1
    if aux == 15:
        break


 

