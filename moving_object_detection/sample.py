import cv2
import imutils as imu

img = cv2.imread("logo.png")   

img = imu.resize(img,400,400,cv2.INTER_AREA)
cv2.imshow("original", img)

bw_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow("black-white", bw_img)

blur_img = cv2.GaussianBlur(bw_img,(15,15),0)
cv2.imshow("Blur img",blur_img)

_,final_img = cv2.threshold(blur_img,50,255,cv2.THRESH_BINARY)
cv2.imshow("final img",final_img)


cv2.waitKey(5000)
cv2.destroyAllWindows()
