import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode =False, maxFaces=2, minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon) # Creating an object
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, Draw= True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
        self.results = self.faceMesh.process(imgRGB)

        faces = []
        # Hand landmarks on each detected hand.
        if self.results.multi_face_landmarks:
            for facelm in self.results.multi_face_landmarks:
                if draw:
                    # Draw landmarks and connections on the face
                    self.mpDraw.draw_landmarks(img, facelm, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                
                face = []
                # Iterate over the face landmarks
                for id, lm in enumerate(facelm.landmark):
                    #print(lm)
                    # Get height, width and channels 
                    ih, iw, ic = img.shape
                    # calculate center
                    x, y = int(lm.x*iw), int(lm.y*ih) 
                    #print(id, x, y)
                    cv2.putText(img, f'{idx}', (x,y), cv2.FONT_HERSHEY_PLAIN, 0.7 ,(0,255,0),1) # shows index of each face landmark 
                    face.append([x,y])
                faces.append(face)
        
        return img, faces 
                

def main():
    # Capturing the video stream
    cap = cv2.VideoCapture(r"C:\Users\gaura\Desktop\A Computer vision\two faces.mp4")
    pTime =0 #previous time
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMesh(img)

        if len(faces)!=0:
            print(len(faces))

        cTime = time.time()
        fps = 1/(cTime-pTime) # calculate FPS
        pTime = cTime

        # Putting text on image
        cv2.putText(img, str(int(fps)), (10,10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)       

        cv2.imshow("Image",img)   
        cv2.waitKey(1)     

if __name__ == "__main__":
    main()