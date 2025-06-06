#Import only if not previously imported
import cv2

vc = cv2.VideoCapture(0)

while True:
    
    _ , img = vc.read()

    cv2.imshow("Video capture ", img)

    key = cv2.waitKey(1)

    if( key == ord('q')):
        break

vc.release()
cv2.destroyAllWindows()

