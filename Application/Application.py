# ACKNOWLEDGMENTS:
# https://shrishailsgajbhar.github.io/post/OpenCV-Apple-detection-counting
# https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=inrange%20function%20with%20the%20range,10%20and%20160%20to%20180.
# https://techtutorialsx.com/2020/11/29/python-opencv-copy-image/
# https://codeloop.org/python-opencv-circle-detection-with-houghcircles/
# https://www.youtube.com/watch?v=8CMTqpZoec8
# https://stackoverflow.com/questions/58353513/detecting-apple-by-thresholding

import cv2 
import numpy as np
import os
import sys

# Save intermediate files
def save(img,thing,name):
    directory_intermediate = os.getcwd() + "/Intermediate"
    if not os.path.exists(directory_intermediate): os.makedirs(directory_intermediate)
    targ_name_out = thing + "_" + name
    targ_path_out = os.path.join(directory_intermediate, targ_name_out)
    cv2.imwrite(targ_path_out, img)

# Function to process images
def process_image(img_in,name):

    fail1 = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    fail2 = cv2.HoughCircles(fail1,cv2.HOUGH_GRADIENT,dp=1.1,minDist=80,param1=100,param2=20,minRadius=20,maxRadius=150)
    fail3 = img_in
    if fail2 is not None:
        col_G = (0,255,0)
        detections	= np.uint16(np.around(fail2))
        num_apl = len(detections[0,:])
        for	index, apples in enumerate(detections[0,:], start=1): 
            cv2.circle(fail3,(apples[0],apples[1]),apples[2],col_G,6)
    save(fail3,"first",name)

    ###  CREATE MASK  ###

    # Blur image and convert to HSV
    ker_siz = 49
    img_blr = cv2.GaussianBlur(img_in, (ker_siz, ker_siz), 0)
    img_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)
    
    save(img_hsv,"hsv",name)

    # Define HSV boundaries
    s_min = 60.0
    v_min = 20.0
    h_max = 80.0
    h_min = 160.0

    # Params for H = MIN -> X
    hsv_A_lo = np.array([  0.0, s_min, v_min])
    hsv_A_hi = np.array([h_max, 255.0, 255.0])
    # Params for H = X -> MAX
    hsv_B_lo = np.array([h_min, s_min, v_min])
    hsv_B_hi = np.array([180.0, 255.0, 255.0])

    # Generate masks based on those params and combine
    mask_A = cv2.inRange(img_hsv, hsv_A_lo, hsv_A_hi)
    mask_B = cv2.inRange(img_hsv, hsv_B_lo, hsv_B_hi)
    mask_fin = mask_A + mask_B

    save(mask_fin,"mask",name)

    ### APPLE DETECTION ###

    # Dialation and Erosion
    ker_siz = 7
    kernel = np.ones((ker_siz,ker_siz), np.uint8)
    #mask_open = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_open = cv2.morphologyEx(mask_fin, cv2.MORPH_OPEN, kernel, iterations=3)
    #mask_open = cv2.erode(mask_final, kernel, iterations=3)
    #mask_open	= cv2.medianBlur(mask_open,	3)
    
    # Detect circular shapes with HoughCircles
    detections	= cv2.HoughCircles(mask_open,cv2.HOUGH_GRADIENT,
                               dp=1.3,
                               minDist=100, 
                               param1=130,
                               param2=20,
                               minRadius=60,
                               maxRadius=130
                               )

    # No apples detetced
    if detections is None: return img_hsv

    # Pre made colours
    col_G = (0,255,0)
    col_W = (255,255,255)
    col_B = (0,0,0)

    # Setup for next phase
    detections	= np.uint16(np.around(detections))
    img_out = img_in#cv2.cvtColor(mask_open, cv2.COLOR_GRAY2BGR)
    num_apl = len(detections[0,:])

    # Highlight each apple
    for	index, apples in enumerate(detections[0,:], start=1):
        cv2.circle(img_out,(apples[0],apples[1]),apples[2],col_G,6) # Draw circle around target
        cv2.circle(img_out,(apples[0],apples[1]),2,col_G,4)         # Draw dot in centre of traget
        cv2.putText(img_out,f"{index}",(apples[0]-20,apples[1]-15), # Put target ID in center of target
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,col_W,2)

    # Print a total count on the image
    cv2.rectangle(img_out, (0,0), (300,40),col_B,cv2.FILLED)        # Background of count
    cv2.putText(img_out,f"Total apples: {num_apl}",(1,30),          # Write total count
                cv2.FONT_HERSHEY_SIMPLEX,1,col_W,2)

    return img_out, num_apl

# Main function
def main():

    # Advise the user that the programme has begun
    print("BEGIN")

    # Declare input and output directories
    directory_input = os.getcwd() + "/Resources"
    directory_output = os.getcwd() + "/Output"
    
    # Throw error if input directory does not exist
    assert os.path.exists(directory_input), "Path error"

    # Create output directory if it does not exist
    if not os.path.exists(directory_output): os.makedirs(directory_output)


    # Run for each image in the input directory
    for targ_name_in in os.listdir(directory_input):

        # Read image in
        targ_path_in = os.path.join(directory_input, targ_name_in)
        targ_image_in = cv2.imread(targ_path_in, cv2.IMREAD_COLOR)

        # Test that image exists
        assert targ_image_in is not None, "Image not found"
        
        # Process image
        targ_image_out, num_apples = process_image(targ_image_in,targ_name_in)

        # Save and display
        targ_name_out = f"Out[{num_apples}]_" + targ_name_in
        targ_path_out = os.path.join(directory_output, targ_name_out)
        cv2.imwrite(targ_path_out, targ_image_out)
        #cv2.imshow(f"{targ_name_out}", targ_image_out)

    # Advise user processing has completed
    print("DONE")

    # Destroy all images at end
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())