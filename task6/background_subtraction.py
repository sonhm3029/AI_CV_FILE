# importing libraries
import numpy as np
import cv2
  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_dil = np.ones((21,21), np.uint8)
  
# creating object
fgbg = cv2.createBackgroundSubtractorMOG2()
  
# capture frames from a camera 
cap = cv2.VideoCapture("C:\\video.mp4")
############
width = int(cap.get(3))
height = int(cap.get(4))
# print(width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
while(1):
    # read frames
    ret, img = cap.read()
    fshape = img.shape
    # img = img[100:fshape[0] - 100,:fshape[1] - 100,: ]
    # print(img.shape[0], img.shape[1])
    #remove noise with gauss filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    # apply mask for background subtraction
    fgmask = fgbg.apply(gray)
    #fgmask = fgbg.apply(img)
      
    # apply transformation to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(fgmask, kernel_dil, iterations= 1)
    
    #find contour
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #draw contour
    for contour in contours:
        (x,y,w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1500:
            continue
        cv2.rectangle(img, (x, y), (x+w, y +h), (0,0,255), 2 )
    # after removing noise
    # img = cv2.flip(img, 0)
    out.write(img)
    cv2.imshow('GMG', img)
      
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
cap.release()
out.release()
cv2.destroyAllWindows()