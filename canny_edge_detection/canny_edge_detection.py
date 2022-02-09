import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    img = cv2.flip(img,1)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg,(5,5),1)
    cannyImg = cv2.Canny(blurImg,200,100)
    kernel = np.ones((2,2))
    imgDial = cv2.dilate(cannyImg,kernel,iterations=2)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)


    #cv2.imshow("wgery images ",grayImg)
    cv2.imshow("blurImg ",blurImg)
    cv2.imshow("cannyImg ",cannyImg)
    cv2.imshow("imgDial ",imgDial)
    cv2.imshow("imgThre ",imgThre)


    if cv2.waitKey(1) & 0xff == 27:
        break
cv2.destroyAllWindows()
cam.release()
