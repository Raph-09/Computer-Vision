import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

goats = []
for name in os.listdir(r'C:\Users\HP\Desktop\datascience_file\opencv\train'):
   goats.append(name)



face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\HP\Desktop\datascience_file\opencv\val\messi _val\ronaldo_n_messi.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('fotballer', gray)


faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 13)

for (x,y,w,h) in faces_rect:
    faces_r = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_r)
    print(f'Label = {goats[label]} with a confidence of {confidence}')

    if label==0:
        cv.putText(img, str(goats[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
    else:
       cv.putText(img, str(goats[label]), (190,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
    
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Goat Face', img)

cv.waitKey(0)