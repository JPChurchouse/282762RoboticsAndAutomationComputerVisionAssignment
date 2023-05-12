# https://gist.github.com/pknowledge/aa1469b7ba8cd652adb652d4359ef4f0
import cv2
import numpy as np
import os
import sys

def nothing(x):
    pass

cv2.namedWindow("a")
cv2.createTrackbar("dp", "a", 1, 50, nothing)
cv2.createTrackbar("mindist", "a", 100, 300, nothing)
cv2.createTrackbar("param1", "a", 130, 300, nothing)
cv2.createTrackbar("param2", "a", 20, 300, nothing)
cv2.createTrackbar("minrad", "a", 50, 500, nothing)
cv2.createTrackbar("maxrad", "a", 200, 500, nothing)


def process_image(img_input):

    # Blur image slightly to reduce noise
    blr_siz = 49
    img_blur = cv2.GaussianBlur(img_input, (blr_siz, blr_siz), 0)

    # Convert to HSV for processing
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    

    # HSV boundaries
    s_min = 60.0
    v_min = 20.0
    h_max = 80.0
    h_min = 160.0

    # lower boundary RED color range values; Hue (0 - 10)
    hsv_A_lo = np.array([ 0.0,  s_min,  v_min]) # Cutoff white
    hsv_A_hi = np.array([h_max, 255.0, 255.0]) # Pure red -> green/
    
    # upper boundary RED color range values; Hue (160 - 180)
    hsv_B_lo = np.array([h_min,  s_min,  v_min]) # Cutoff white
    hsv_B_hi = np.array([180.0, 255.0, 255.0]) # Magenta/ -> Red

    # Run maskwork
    mask_A = cv2.inRange(img_hsv, hsv_A_lo, hsv_A_hi)
    mask_B = cv2.inRange(img_hsv, hsv_B_lo, hsv_B_hi)
    
    mask_cutout = mask_A + mask_B


    kernel = np.ones((21,21),np.uint8)
    mask_shrink = cv2.morphologyEx(mask_cutout, cv2.MORPH_OPEN,kernel)

    img_overlay = cv2.bitwise_and(img_input, img_input, mask=mask_shrink)

    # canny edges, make cirlces

    return mask_shrink


def process2(img_in):
    img_grey = process_image(img_in)
    #img_grey	=	cv2.cvtColor(img_in,	cv2.COLOR_BGR2GRAY)
    img_blur	= cv2.medianBlur(img_grey,	5)
    cv2.imshow("grey", img_grey)
    
    circles	= cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,
                               dp=cv2.getTrackbarPos("dp","a"),
                               minDist=cv2.getTrackbarPos("mindist","a"),
                               param1=cv2.getTrackbarPos("param1","a"),
                               param2=cv2.getTrackbarPos("param2","a"),
                               minRadius=cv2.getTrackbarPos("minrad","a"),
                               maxRadius=cv2.getTrackbarPos("maxrad","a"))
    
    if circles is None: return img_in
    circles	= np.uint16(np.around(circles))
    img_out = img_in.copy()
    for	i in circles[0,:]:
        #	draw	the	outer	circle
        cv2.circle(img_out,(i[0],i[1]),i[2],(0,255,0),6)
        #	draw	the	center	of	the	circle
        cv2.circle(img_out,(i[0],i[1]),2,(0,0,255),3)
    
    return img_out


while True:
    frame = cv2.imread(os.getcwd() + "/Resources/9.png")

    res = process2(frame)

    cv2.imshow("frame", frame)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()