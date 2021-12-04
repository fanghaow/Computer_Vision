import cv2
import sys
import numpy as np

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#capture frame by frame
filepath = "input/"
pNum = 40
faceNum = 5
breakFlag = False
for id in range(1, pNum+1):
    for face in range(1, faceNum+1):
        if face == 10:
            file = filepath + str(id) + "/" + str(face) + ".png"
        else:
            file = filepath + str(id) + "/0" + str(face) + ".png"

            frame = cv2.imread(file, 0)
            if frame is None:
                print("[ERROR]: Load training images failed! Please check your path.\n")
                breakFlag = True
                break
            # cv2.imshow("Origin", frame)
            # cv2.waitKey(0)

            gray = frame.copy() # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray) #, scaleFactor=1.1, minNeighbors= 5)

            #Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 2)

            newImg = gray[y:y+h, x:x+w]
            cv2.imshow("Face", newImg)

            cv2.imshow('Face Position',frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                breakFlag = True
                break
    
    if breakFlag == True:
        break

cv2.destroyAllWindows()