import imutils as imu
import cv2

cam = cv2.VideoCapture(0)

firstFrame = None
area = 400

while True:
    
    _,img = cam.read()
    
    text = 'Normal'
    
    img = imu.resize(img,width=500)
    
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(21,21),0)
    
    if firstFrame is None:
        firstFrame = blur_img
        continue
    
    diffImg = cv2.absdiff(firstFrame,blur_img)
    
    thresh_diffImg = cv2.threshold(diffImg,25,255,cv2.THRESH_BINARY)[1]
    thresh_diffImg = cv2.dilate(thresh_diffImg,None,iterations=2)
    
    cnts = cv2.findContours(thresh_diffImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imu.grab_contours(cnts)
    
    for c in cnts:
        if( cv2.contourArea(c) <= 400):
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        text = 'Moving object detected'
    
    cv2.putText(img,text,(20,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    print(text)
    
    cv2.imshow("Moving object Detection",img)
    
    
    key = cv2.waitKey(10)
    
    if( key == ord('q')):
        break
    
cam.release()
cv2.destroyAllWindows()   