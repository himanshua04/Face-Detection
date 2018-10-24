import cv2
import numpy as np
import os
facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
samplenum=0
uid=input('enter id')
cam=cv2.VideoCapture(0)
if not os.path.exists('proof'):
    os.makedirs('proof')

while(True):
    
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces: 
        samplenum+=1
        cv2.imwrite('dataset/'+str(uid)+'_'+str(samplenum)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    
    cv2.imshow('face',img)
    cv2.waitKey(1)
    if(samplenum>50):
        break

cam.release()
cam.destroyAllWindows()

        
