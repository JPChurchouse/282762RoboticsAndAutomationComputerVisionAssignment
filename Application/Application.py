import cv2 
import numpy as np

import os
import sys


def main():

    ## Declare input and output directories
    directory_input = os.getcwd() + "/Resources"
    directory_output = os.getcwd() + "/Output"

    ## Throw error if input directory does not exist
    if not os.path.exists(directory_input):
        return 1

    ## Create output directory if it does not exist
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)


    # Run for each image in the input directory
    for targ_name_in in os.listdir(directory_input):

        # Generate local file variables
        targ_path_in = os.path.join(directory_input, targ_name_in)
        targ_image_in = cv2.imread(targ_path_in, cv2.IMREAD_COLOR)

        # Test that image exists
        if (targ_image_in is None):
            print(f'Error: Image not found')
            return 1
        
        # Process image
        targ_image_out = process_image(targ_path_in)

        # Save and display
        targ_name_out = targ_name_in
        targ_path_out = os.path.join(directory_output, targ_name_out)
        cv2.imwrite(targ_path_out, targ_image_out)
        
        cv2.imshow(f"{targ_name_out}", targ_image_out)


    # Destroy all images at end
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())


# Function to process images
def process_image(image_input):
    num_apples = 0

    ###
    range_low = np.array([1,2,3])
    range_hig = np.array([1,2,3])


    ###

    cv2.rectangle(image_hsv, (0,0), (32, 18), (255,255,255), -1)#create rectangle in corner
    cv2.putText(image_hsv, f"{num_apples}", (1,1), cv2.FONT_HERSHEY_COMPLEX, 0.2, (0,0,0),1,1)
    image_hsv = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV)
    return image_hsv



