import cv2
import numpy as np
import time

import PoseEstimation_module as pm

cap = cv2.VideoCapture(r"C:\Users\gaura\Desktop\A Computer vision\Workout.mp4")

detector = pm.poseDetector()

count = 0
dir = 0
pTime = 0
while True:
    #read the image
    img = cv2.imread(r"C:\Users\gaura\Desktop\A Computer vision\Exercise.jpg")
    img = detector.findPose(img) # find the pose
    lmList = detector.findPosition(img) # find the position
    # if lmList is not empty
    if len(lmList):
        # Right arm (the values are points on arm from mediapipe documentation)
        angle=detector.findAngle(img,12,14,16)
        # Left Arm
        detector.findAngle(img, 11, 13, 15)
        # Converting range of values to 0-100
        per = np.interp(angle, (210, 310), (0,100))
        bar = np.interp(angle, (220, 310), (658,100))

        # Check for dumbell curls
        color = (255,0,255)
        if per == 100:
            color = (0,255,0)
            if dir == 0:
                count += 0.5
                dir = 1

        if per == 0:
            color = (0,255,0)
            if dir == 1:
                count += 0.5
                dir = 0        
        # Draw Bar
        cv2.rectangle(img, (1100,100), (1175,650), color, 3)
        cv2.rectangle(img, (1100,int(bar)), (1175,650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Curl
        cv2.rectangle(img, (0,450), (250,720), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,(255, 0, 0), 25)

    cTime = time.time() # Current time
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)
    cv2.imshow('Image', img)
    cv2.waitKey(1)