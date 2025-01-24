#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from config import load_config

def main():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    # TODO: Need to setup in another place
    carpeta=f'{current_dir}/DATASET-UNCINARIAS'

    json_file_path = sys.argv[1]
    config = load_config(json_file_path) 
    print('config', config)
    print('config is none?', config is None)
    print('config.CUTTINGS_NEGATIVES', config.CUTTINGS_NEGATIVES)

    names=os.listdir(carpeta)
    cont=1
    path = f'{config.CUTTINGS_NEGATIVES}'
    path2 = f'{config.CUTTINGS_POSITIVES}'
    print('path', path)
    print('path2', path2)

    for i in names:
        print('- name', i)
        imagen_direccion=f'{carpeta}/{i}'
        img =cv2.imread(imagen_direccion)
        copia=img
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray =cv2.resize(img_gray, (500,500),interpolation=cv2.INTER_LANCZOS4)

        copia=cv2.resize(copia,(500,500),interpolation=cv2.INTER_LANCZOS4)
        ### cv2.imshow('copia',copia)

        #proceso de reconocimiento
        bilateralFilter=cv2.bilateralFilter(img_gray, 9, 75, 75)
        ### cv2.imshow('bilateralFilter',bilateralFilter)
        #bilateralFilter = cv2.bilateralFilter(img_gray, 9, 75, 75)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(bilateralFilter,kernel,iterations =3)
        ### cv2.imshow('erosion',erosion)
        dilation = cv2.dilate(erosion,kernel,iterations =2)
        ### cv2.imshow('dilation',dilation)
        _, frame5 = cv2.threshold(dilation,1,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ### cv2.imshow('frame5',frame5)
        kernel2 = np.ones((20, 20), np.uint8)
        #kernel3 = np.ones((10, 10), np.uint8)
        #closing = cv2.morphologyEx(frame5, cv2.MORPH_HITMISS, kernel2)
        closing = cv2.morphologyEx(frame5, cv2.MORPH_DILATE, kernel2)
        ### cv2.imshow('MORPH_DILATE',closing)
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
        ### cv2.imshow('MORPH_CLOSE',closing)

        canny=cv2.Canny(closing,50,180,None,3)
        ### cv2.imshow('canny',canny)
        

        (c,j)=cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in c:
            m=cv2.moments(i)
            print("variable " +str(i) + " :" +str(m['m00']))
            if m["m00"] != 0:
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
            else:
                # set values as what you need in the situation
                cX, cY = 0, 0

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
            recorte=cv2.resize(recorte, (500,500),interpolation=cv2.INTER_LANCZOS4)

            if m["m00"] >= 1000:
                cv2.imwrite(os.path.join(path2 , f'{config.CUTTINGS_FILE_PREFIX}'+str(cont)+f'{config.CUTTINGS_CUT_EXTENSION}'), recorte)
            else :
                cv2.imwrite(os.path.join(path , f'{config.CUTTINGS_FILE_PREFIX}'+str(cont)+f'{config.CUTTINGS_CUT_EXTENSION}'), recorte)

            ## ONLY FOR WINDOWS OS
            # if cv2.waitKey(0):
            cont+=1    

if __name__ == "__main__":
    print('------ 1. CUTTINGS START ------')
    main()
    print('------ CUTTINGS END ------')