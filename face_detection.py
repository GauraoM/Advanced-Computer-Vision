import cv2
import mediapipe as mp
import time

# Capturing the video stream
cap = cv2.VideoCapture(r"C:\Users\gaura\Desktop\A Computer vision\face.mp4")

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection() # Creating an object
mpDraw = mp.solutions.drawing_utils

pTime =0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
    results = faceDetection.process(imgRGB)

   # Hand landmarks on each detected hand.
    if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(image, detection)
                # print(id,detection)
                # Store the location of points in a bboxC and use it
                bboxC = detection.location_data.relative_bounding_box

                # Get image parameters
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                      int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw box over face
                cv2.rectangle(img, bbox, (255,0,255), 2)
                cv2.putText(img, str(f'{int(detection.score[0]*100)}%'), (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                            (255,0,255), 2)


    cTime = time.time()
    fps = 1/(cTime-pTime) # calculate FPS
    pTime = cTime

    # Putting text on image
    cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)       

    cv2.imshow("Image",img)   
    cv2.waitKey(1)     
