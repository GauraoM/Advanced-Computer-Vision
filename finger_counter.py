import os
import time

import cv2
import mediapipe as mp
import HandTrackingModule as htm

wCam, hCam = 648, 488

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Open the webcam
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0 # Previous time

detector = htm.handDetector(min_detection_confidence = 0.7)

# List the directory where all finger images stored
folderPath = "Fingerprints"
myList = os.listdir(folderPath)
print(myList)

# Overlay list to overlay on our webcam image
overlayList = []
# Iterating over each image in the folder and read them
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # append it to overlaylist
    overlayList.append(image)
    
    
detector = htm.handDetector(max_num_hands=1, min_detection_confidence=0.75)

tipIds = [4,8,12,16,20] # Tip points of the finger  
while True:
    # Reading the frames
    success, img = cap.read()

    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False) 
    #print(lmList)

    # if length of lmlist is not equals to zero
    if len(lmList) != 0:
        fingers = []

        # For Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for other fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1) # it's open
            else:
                fingers.append(0)   

        #print(fingers)
        # No. of 1s present
        total_fingers = fingers.count(1)
        print(total_fingers) 

        h, w, c = overlayList[total_fingers-1].shape
        img[0:h, 0:w] = overlayList[total_fingers-1]  
        
        cv2.rectangle(img, (10,460), (150, 600), (0.255,0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    cTime = time.time() # Current time
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img ,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)