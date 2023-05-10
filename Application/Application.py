import cv2 
import numpy as np

import os
import sys

# remove white
hsv_low_white = np.array([0,52,0])
hsv_high_white = np.array([255,255,255])

# 



# Function to process images
def process_image(image_input):
    grey = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(grey, (11, 11), 0)
    circles	= cv2.HoughCircles(grey,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=0,maxRadius=0)
    circles	= np.uint16(np.around(circles))

    for i in circles[0,:]:
        cv2.circle(image_input, (i[0], i[1]), i[2], (0,255,0), 2)
        
    
    
    return image_input




def main():

    ## Declare input and output directories
    directory_input = os.getcwd() + "/Resources"
    directory_output = os.getcwd() + "/Output"

    ## Throw error if input directory does not exist
    assert os.path.exists(directory_input), "Path error"

    ## Create output directory if it does not exist
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)


    # Run for each image in the input directory
    for targ_name_in in os.listdir(directory_input):

        # Generate local file variables
        targ_path_in = os.path.join(directory_input, targ_name_in)
        targ_name_out = targ_name_in
        targ_path_out = os.path.join(directory_output, targ_name_out)
        
        # Read image in
        targ_image_in = cv2.imread(targ_path_in, cv2.IMREAD_COLOR)

        # Test that image exists
        assert targ_image_in is not None, "Image not found"
        
        # Process image
        targ_image_out = process_image(targ_image_in)

        # Save and display
        cv2.imwrite(targ_path_out, targ_image_out)
        cv2.imshow(f"{targ_name_out}", targ_image_out)


    # Destroy all images at end
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())