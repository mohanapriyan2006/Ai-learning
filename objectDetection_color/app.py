import cv2
import imutils as imu

inLower= (0,84,182)
inUpper = (179,255,255)

vc = cv2.VideoCapture(0)

print("Program is running.... \n To Stop this Program, Click 'q'")

while True:
    hasCam,img = vc.read()
    
    if not hasCam:
        print("Error Occured !")
        break
    
    text1 = "Orange Object is Not Detected !"
    text1_clr = (0,255,200)
    
    text2 = "Normal"
    text2_clr = (255,255,0)
    
    img = imu.resize(img,width=500)
    blur_img = cv2.GaussianBlur(img,(11,11),0)
    hsv_img = cv2.cvtColor(blur_img,cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_img,inLower,inUpper)
    mask = cv2.erode(mask,None,iterations=2)
    mask = cv2.dilate(mask,None,iterations=2)
    
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    center = None
    
    if(len(cnts) > 0):
            c = max(cnts , key=cv2.contourArea)
            (x,y) , radius = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"]/ M["m00"]) , int(M["m01"]/M["m00"]))
                
            if(radius > 10):
                cv2.circle(img,(int(x) , int(y)) , int(radius) , (0,255,255) , 2)
                cv2.circle(img,center, 4 , (0,0,255) , -1)
                text1 = "Orange Object is Detected."
                text1_clr = (0,255,0)
                
                if(radius>180):
                    text2 = "plz stop!"
                    text2_clr = (255,0,0)
                elif(center[0] < 150):
                    text2 = "Go Rigth"
                elif(center[0] > 450):
                    text2 = "Go Left"
                else:
                    text2 = "Come Front"
                    
    cv2.putText(img,text1,(2,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text1_clr,2)
    cv2.putText(img,text2,(350,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,text2_clr,2)
    
    cv2.imshow("Orange color Object Detector",img)
    
    
    key = cv2.waitKey(10)
    if(key == ord('q') or key == ord('Q')):
        print("Program was stoped!")
        break
    
vc.release()
cv2.destroyAllWindows()
    