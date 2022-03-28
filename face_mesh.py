import cv2
import mediapipe as mp
import time

# Capturing the video stream
cap = cv2.VideoCapture(r"C:\Users\gaura\Desktop\A Computer vision\two faces.mp4")

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # Creating an object
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

pTime =0 #previous time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
    results = faceMesh.process(imgRGB)

   # Hand landmarks on each detected hand.
    if results.multi_face_landmarks:
        for facelm in results.multi_face_landmarks:
            # Draw landmarks and connections on the face
            mpDraw.draw_landmarks(img, facelm, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            # Iterate over the face landmarks
            for id, lm in enumerate(facelm.landmark):
                #print(lm)
                # Get height, width and channels 
                ih, iw, ic = img.shape
                # calculate center
                x, y = int(lm.x*iw), int(lm.y*ih) 
                print(id, x, y) 
            

    cTime = time.time()
    fps = 1/(cTime-pTime) # calculate FPS
    pTime = cTime

    # Putting text on image
    cv2.putText(img, str(int(fps)), (10,10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)       

    cv2.imshow("Image",img)   
    cv2.waitKey(1)     
