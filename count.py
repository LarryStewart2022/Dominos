import cv2
#import numpy as np
#from rembg import remove
#from PIL import Image
#import io


def detect_pips(uploaded_image):
    if uploaded_image is None:
        return None
    image = cv2.resize(uploaded_image, (512, 512))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(gray_image)
    
    # Number of pips is the number of keypoints
    pip_count = len(keypoints)
    return pip_count


uploaded_image = cv2.imread('28.png')

pip_count = detect_pips(uploaded_image)
print(f'Total pip count is: {pip_count}')