import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushthickness = 15
eraserthickness = 50
#######################

# Going through images 
folderpath = "header"
myList = os.listdir(folderpath)

# Iterating over the images and append it to overlayList
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(min_detection_confidence=0.70)
xp, yp =0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    # read the frames
    success, img = cap.read()
    # flip the image
    img = cv2.flip(img, 1)
    # find landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, 0, False)

    if len(lmList)!=0:

        #Tip of the index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

        # If selection mode- Two fingers are up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor,cv2.FILLED)
            print("selection mode")
            # Checking for click
            if y1 < 125:
                if 250 <=x1<= 450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                else:
                    header = overlayList[3] 
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)           

         # If draw mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")

            if xp == 0 and yp == 0:
                 xp, yp = x1, y1

            if  drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserthickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserthickness)
            else:
            
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushthickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushthickness)
            xp, yp = x1, y1

    # Overlay canvas image on original one
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) # convert to gray
    _, imgInv = cv2.threshold(imgGray, 550, 255, cv2.THRESH_BINARY_INV) # inverting the image
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR) # Conveting inverse gray image to bgr  

    # Bitwise operations
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting header image
    img[0:116, 0:1193] = header
    # Adding two images
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5,0)
    cv2.imshow('Image',img)
    cv2.imshow('Canvas',imgCanvas)
    cv2.waitKey(1)