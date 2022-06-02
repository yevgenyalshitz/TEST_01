import cv2
import numpy as np
import face_recognition
import os

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#Step(1) Get source images:
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
# print("\n".join(myList))

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
# print("\n".join(classNames))

#Step(2) Encode source images:
encodeListKnown = findEncodings(images)
print('Encoding Completed for the following', len(encodeListKnown), 'images:\n', "\n".join(classNames))

#Step(3) Get comparison images from the webcam
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()

    #resize all images by 0.25
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

#Step(4) Find matches
    # grab encodeFace from encodeCurFrame & faceLoc from faceCurFrame
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print('\n'.join(faceDis))
        print(faceDis)





    # faceLoc = face_recognition.face_locations(imgElon)[0]

    # cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
        #
        # faceLocTest = face_recognition.face_locations(imgTest)[0]
        # encodeTest = face_recognition.face_encodings(imgTest)[0]
        # cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
        #
        # # print(faceLoc)
        #
        # # tolerance – How much distance between faces to consider it a match. Lower is more strict. 0.6 (euclidean distance) is typical best performance.
        # results = face_recognition.compare_faces([encodeElon], encodeTest)
        #
        # faceDis = face_recognition.face_distance([encodeElon], encodeTest)

