import cv2
import numpy as np

img=cv2.imread('Ascaris lumbricoides_0002.jpg')

copia=img

img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

img=cv2.resize(img,(500,500),interpolation=cv2.INTER_AREA)
copia=cv2.resize(hist_equalization_result,(500,500),interpolation=cv2.INTER_AREA)



#recorte=img[60:220,280:480]
#recorte=cv2.resize(recorte,(500,500),interpolation=cv2.INTER_LANCZOS4)




cv2.imshow('Original',img)
cv2.imshow('Copia',copia)
#cv2.imshow('Recorte',recorte)

cv2.waitKey(0)
cv2.destroyAllWindows()

