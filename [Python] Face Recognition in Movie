#"Vincenzo" is the hottest 2021 South Korean Netflix TV series
#This project is to use Face Recognition is count how many times Vincenzo (Song Joong Ki) appear in the trailer.

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#custom file name. 
#Open a file named "People". It includes different people names and their images. 
#This is the image URL: https://img.i-scmp.com/cdn-cgi/image/fit=contain,width=1098,format=auto/sites/default/files/styles/1200x800/public/d8/images/canvas/2021/02/18/404ed5bb-de13-49fc-8603-95f19ac376cf_f67f50f3.png?itok=vBVJgt3T&v=1613620269
#Please save the image and put it into the file called "People"

#This is my personal computer file path
path = 'C:/Users/User/FaceRecognitionProject/People'

images =[]
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('number_of_people_appeared','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
    
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture('vincenzo.mp4')

while True:
    ret, img = cap.read()
    #imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(gray)
    encodesCurFrame = face_recognition.face_encodings(gray,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    
# Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
