from rembg import remove
from PIL import Image
import cv2
import numpy as np
import io


def detect_pips(uploaded_image):
    if uploaded_image is None:
        return None
    
    # Convert the numpy array image to bytes
    is_success, im_buf_arr = cv2.imencode(".png", uploaded_image)
    byte_im = im_buf_arr.tobytes()

    # Remove the background
    output_bytes = remove(byte_im)

    # Convert the output bytes to an image
    output_image = Image.open(io.BytesIO(output_bytes))

    # Convert the PIL Image back to a numpy array
    image = np.array(output_image)

    # Resize the image
    image = cv2.resize(image, (512, 512))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area
    params.filterByArea = True
    params.minArea = 50

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(gray_image)
    
    # Number of pips is the number of keypoints
    pip_count = len(keypoints)
    return pip_count



# Load an image from file
uploaded_image = cv2.imread('46.jpg')

# Call the function
pip_count = detect_pips(uploaded_image)

# Output the result
print(f'Total pip count is: {pip_count}')