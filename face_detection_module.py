import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectioncon=0.5):
        self.minDetectioncon = minDetectioncon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectioncon) # Creating an object
        self.mpDraw = mp.solutions.drawing_utils

    def findfaces(self, img,draw = True ):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
    # Hand landmarks on each detected hand.
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(image, detection)
                # print(id,detection)
                # Store the location of points in a bboxC and use it
                bboxC = detection.location_data.relative_bounding_box

                # Get image parameters
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                      int(bboxC.width * iw), int(bboxC.height * ih)

                bboxes.append([bbox,detection.score[0]])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    # Put detection score on image
                    cv2.putText(img, str(f'{int(detection.score[0]*100)}%'), (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                                (255,0,255), 2)
        
        return img, bboxes # returns image and bounding boxes around face

    def fancyDraw(self, img, bbox,l=30, t=5, rt=1 ):
        # l=length, t=thickness, rt=rectangle thickness
        x,y,w,h = bbox
        x1, y1 = x + w, y + h # Lower right corner

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Drawing line along the corners(top left)
        cv2.line(img, (x,y), (x+l,y), (255,0,255), t)
        cv2.line(img, (x,y), (x,y+l), (255,0,255), t)

        # Drawing line along the corners(top right)
        cv2.line(img, (x1,y), (x1-l,y), (255,0,255), t)
        cv2.line(img, (x1,y), (x1,y+l), (255,0,255), t)

        # Drawing line along the corners(bottom left)
        cv2.line(img, (x,y1), (x+l,y1), (255,0,255), t)
        cv2.line(img, (x,y1), (x,y1-l), (255,0,255), t)

        # Drawing line along the corners(bottom right)
        cv2.line(img, (x1,y1), (x1-l,y1), (255,0,255), t)
        cv2.line(img, (x1,y1), (x1,y1-l), (255,0,255), t)
        
        return img

def main():
    # Capturing the video stream
    cap = cv2.VideoCapture(r"C:\Users\gaura\Desktop\A Computer vision\two faces.mp4")

    pTime =0
    detector = FaceDetector()
    while True:
        success, img = cap.read()

        img, bboxes = detector.findfaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime) # calculate FPS
        pTime = cTime

        # Putting text on image
        cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)       

        cv2.imshow("Image",img)   
        cv2.waitKey(1)     


if __name__ == "__main__":
    main()