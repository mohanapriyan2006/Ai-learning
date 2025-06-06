import cv2

algr = 'haarcascade_frontalface_default.xml'

haar = cv2.CascadeClassifier(algr)

cam = cv2.VideoCapture(0)

# cam = cv2.VideoCapture('video.mp4')

# img = cv2.imread('image.png')

print("Program is running ...")

while True:
    
    notError,img = cam.read()
    
    if not notError:
        print('Error occured !')
        break
    
    text = 'Face Not Detected !'
    font_color = (0,0,255)
    
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = haar.detectMultiScale(gray_img,1.3,3)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x + w, y + h),(255,0,0),2)
        text = 'Face Detected :)'
        font_color = (0,255,0)
        
    cv2.putText(img,text,(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,font_color,2)
    
    cv2.imshow("Face Detection",img)
    
    key = cv2.waitKey(10)
    
    # ESC button
    if( key == 27):
        break
    
print('Program is closed !')
cam.release()
cv2.destroyAllWindows()