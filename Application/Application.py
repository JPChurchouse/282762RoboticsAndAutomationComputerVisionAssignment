import cv2 
import numpy as np

import os
import sys



def detect_apples(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds of red color (apples)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Define lower and upper bounds of green color (apples)
    lower_green = np.array([40,50,50])
    upper_green = np.array([80,255,255])
    mask3 = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks to obtain final mask
    mask = cv2.bitwise_or(mask1, mask2, mask=np.zeros_like(mask1))
    mask = cv2.bitwise_or(mask3, mask, mask=np.zeros_like(mask3))

    # Apply morphological transformations to mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected apples
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    # Show image with bounding boxes around detected apples
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





    

def main():

    ## Declare input and output directories
    input_dir = os.getcwd() + "/Resources"
    output_dir = os.getcwd() + "/Output"

    ## Throw error if input directory does not exist
    if not os.path.exists(input_dir):
        return 1

    ## Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Run for each image in the input directory
    for file in os.listdir(input_dir):

        # Generate local file variables
        input_filepath = os.path.join(input_dir, file)
        input_image = cv2.imread(input_filepath, cv2.IMREAD_COLOR)

        # Test that image exists
        if (input_image is None):
            print(f'Error: Image not found')
            return 1
        
        detect_apples(input_filepath)
        # Process image
        #output_image = process_image(input_filepath)

        # Save and display
        #cv2.imshow(f"{file}", output_image)
        #cv2.imwrite(output_dir + f"/_{file}",output_image)


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



