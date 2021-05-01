import os
import cv2 as cv
import numpy as np


goats = []
for name in os.listdir(r'C:\Users\HP\Desktop\datascience_file\opencv\train'):
   goats.append(name)

DIR = r'C:\Users\HP\Desktop\datascience_file\opencv\train'

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = []
labels = []

def face_train():
    for goat in goats:
        path = os.path.join(DIR, goat)
        label = goats.index(goat)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_cord = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_cord:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


face_train()
print('done')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the Recognizer on the features and labels
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)