import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,self.min_detection_confidence,
            self.min_tracking_confidence)

        self.tipIds = [4,8,12,16,20]    

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting to RGB
        self.results = self.hands.process(imgRGB)

         # Hand landmarks on each detected hand.
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw = True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        # Check whether multi hand landmarks are available
        if self.results.multi_hand_landmarks:
            # Give the hand no
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                
                # Get height, width and channels 
                h, w, c = img.shape
                # calculate center
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy) 
                #print(id, cx, cy)
                # Append them to list
                self.lmList.append([id,cx,cy])
                if draw:
                    if id==4:
                        cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)  

        return self.lmList, bbox         
    
    def fingersUp(self):

        fingers = []

        # For Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for other fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1) # it's open
            else:
                fingers.append(0)   

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
    
        return length, img, [x1, y1, x2, y2, cx, cy]    
        

def main():

     # Capturing the video stream
    video_cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    detector = handDetector()

    pTime =0
    cTime =0

    while True:
        success, img = video_cap.read()
        # Once image detected give it to
        img = detector.findHands(img)
        # Get the list of positions
        # lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime) # calculate FPS
        pTime = cTime

        # Putting text on image
        cv2.putText(img, str(int(fps)), (10,10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)       

        cv2.imshow("Image",img)   
        cv2.waitKey(1)  

if __name__=='__main__':
    main()       


