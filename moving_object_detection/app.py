import imutils as imu
import cv2
import time


cam = cv2.VideoCapture(0)
time.sleep(2)  

firstFrame = None
area = 500  

for _ in range(30):
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture initial frame.")
        break
    frame = imu.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    firstFrame = blur

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to read from camera.")
        break

    text = 'Normal'
    text_color = (0, 255, 0)

    img = imu.resize(img, width=500)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    diffImg = cv2.absdiff(firstFrame, blur_img)
    thresh_diffImg = cv2.threshold(diffImg, 25, 255, cv2.THRESH_BINARY)[1]
    thresh_diffImg = cv2.dilate(thresh_diffImg, None, iterations=2)

    cnts = cv2.findContours(thresh_diffImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imu.grab_contours(cnts)

    motion_detected = False
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = 'Moving object detected'
        text_color = (0, 0, 255)
        motion_detected = True

    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)
    print(text)

    cv2.imshow("Motion Detection", img)

    if not motion_detected:
        firstFrame = blur_img.copy()

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
