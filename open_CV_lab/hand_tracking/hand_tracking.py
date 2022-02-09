import cv2
import mediapipe as mp
import time
import autopy

cTime , pTime = 0, 0

##########################
wCam, hCam = 1080, 1920
frameR = 100  # Frame Reduction
smoothening = 7
cx4,cy4 = 0,0
cx8,cy8 = 0,0
#########################

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # to enumerate and print all the point of the hands
            for id , lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy  = int(lm.x*w), int(lm.y*h)

                if id == 4:
                    cx4,cy4 = cx,cy
                if id == 8:
                    cx8,cy8 = cx,cy

                for i in range(id):
                    #cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(0,200,0),2)
                    if id == 8 or id ==4:
                        cv2.circle(img,(cx,cy),10,(255,0,255),2)
                        cv2.line(img,(cx4,cy4),(cx8,cy8),(255,0,0),2)


            mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cTime=time.time()
    fps = 1 / (cTime-pTime)

    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(250,0,250),3)

    cv2.imshow("hand",img)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
