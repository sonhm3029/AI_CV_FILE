from typing import Coroutine
import cv2 as cv
import numpy as np

cap = cv.VideoCapture("C:\\video.mp4")

#check if camera is opened successfully
if not cap.isOpened():
    print("Error in opening video file")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

#Read until video is completed
while ( cap.isOpened()):
    # if ret == True:
        #get difference from 2 frame
        diff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        #apply gaussian blur to remove noise
        blur = cv.GaussianBlur(gray, (5,5), 0)
        _, thres = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thres, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x,y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < 1000:
                continue
            cv.rectangle(frame1, (x, y), (x +w, y+ h), (0, 255, 0), 2)
        #Display the resulting frame
        cv.imshow('Frame', frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        #Press Q to exit
        if cv.waitKey(40)  & 0xFF == ord('q'):
            break

    # else:
    #     break

cap.release()
cv.destroyAllWindows()