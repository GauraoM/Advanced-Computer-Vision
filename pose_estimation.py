import cv2
import mediapipe as mp
import time

# Capturing the video stream
cap = cv2.VideoCapture("C:\\Users\\gaura\\Desktop\\A Computer vision\\pexelscampus.mp4")

mpPose = mp.solutions.pose
pose = mpPose.Pose() # Creating an object
mpDraw = mp.solutions.drawing_utils

pTime =0
cTime =0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
    results = pose.process(imgRGB)

   # Hand landmarks on each detected hand.
    if results.pose_landmarks:
            mpDraw.draw_landmarks(img,results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                #print(id,lm)
                # Get height, width and channels 
                h, w, c = img.shape
                # calculate center
                cx, cy = int(lm.x*w), int(lm.y*h) 
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)   
            # Draw landmarks and connections on the image
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime) # calculate FPS
    pTime = cTime

    # Putting text on image
    cv2.putText(img, str(int(fps)), (10,10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)       

    cv2.imshow("Image",img)   
    cv2.waitKey(1)     
