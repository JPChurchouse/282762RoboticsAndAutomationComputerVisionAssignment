# Name: approach_2.py
# By Dr. S. S. Gajbhar
import cv2 
import numpy as np 
import argparse 

import os
import sys

def main():

    # Defining the color ranges to be filtered.
    # The following ranges should be used on HSV domain image.
    low_apple_red = (160.0, 153.0, 153.0)
    high_apple_red = (180.0, 255.0, 255.0)
    low_apple_raw = (0.0, 150.0, 150.0)
    high_apple_raw = (15.0, 255.0, 255.0)

    input_dir = os.getcwd() + "/Resources"
    output_dir = os.getcwd() + "/Output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):

        filepath = os.path.join(input_dir, file)
        image_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)

        if (image_bgr is None):
            print(f'Error: Image not found')
            return 1
        

        image = image_bgr.copy()
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        mask_red = cv2.inRange(image_hsv,low_apple_red, high_apple_red)
        mask_raw = cv2.inRange(image_hsv,low_apple_raw, high_apple_raw)

        mask = mask_red + mask_raw


        cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        c_num=0
        for i,c in enumerate(cnts):
            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            if r>34:
                c_num+=1
                cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.putText(image, "#{}".format(c_num), (int(x) - 10, int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                continue

        #cv2.imshow("Original image", image_bgr)
        cv2.imshow(f"{file}", image)
        #cv2.imshow("HSV Image", image_hsv)
        #cv2.imshow("Mask image", mask)
        cv2.imwrite(output_dir + f"/_{file}",image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    sys.exit(main())

