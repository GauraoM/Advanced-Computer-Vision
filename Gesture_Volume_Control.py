import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
wCam, hCam = 640, 480
#############################

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Open the webcam
# Setting camera width and height
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0 # Previous time

detector = htm.handDetector(min_detection_confidence = 0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()


# Set minimum and maximum volume range
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0
while True:
    # Reading the frames
    success, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # If length of lmList is not zero
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])

        # storing the points to the variables
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Get the centre between points
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # Draw a circle around them
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
        # Create a line between them
        cv2.line(img, (x1,y1), (x2,y2),(255,0,255), 3)
        # Draw cicle around center of line
        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1) # calculates Hypoteneous of right angle

        # Hand range = 50-300
        #Volume Range -65 -0
        # Gives piecewise linear interpolation
        vol = np.interp(length, [50,300],[minVol, maxVol])
        print(vol)
        # Volume bar
        volBar = np.interp(length, [50,300], [400,150])
        # Volume percentage
        volPer = np.interp(length, [50,300], [0,100])
            
        volume.SetMasterVolumeLevel(vol, None)

        # If length is less than 50 then change the color
        if length< 50:
            cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (0,255,0),3)
    cv2.rectangle(img, (50, int(volBar)), (85,400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f'FPS{int(volPer)}%', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),3)

    cTime  = time.time() # Current Time
    fps = 1/ (cTime- pTime) # Frames/sec
    pTime = cTime
     #write text on image
    cv2.putText(img, f'FPS{int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),3)
    cv2.imshow("Img", img)
    cv2.waitKey(1)