# Smart-Camera-
"""
The solution is to make a camera that takes attendance  and absence automatically when a person passes  through the gate, it takes it through the lens of the eye,  and the data received to him that I learn, gram on it through codes, and to increase security,.
"""

import cv2
import numpy as np
import face_recognition
import os
# Existing code remains the same...
path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)
print(personsList)
for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
print('Encoding Complete.')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to load the image.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    for encodeFace, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.50:  # Confidence threshold
            name = classNames[matchIndex].upper()
            confidence = round((1 - faceDis[matchIndex]) * 100, 2)
        else:
            name = "Unknown"
            confidence = round((1 - min(faceDis)) * 100, 2) if faceDis.size > 0 else 0

        print(f"{name}, Confidence: {confidence}%")

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)

    # In this code creat a two function only now .
