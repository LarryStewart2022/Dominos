import cv2
import numpy as np
from rembg import remove
import io
import os
from PIL import Image


def detect_pips(uploaded_image):
    if uploaded_image is None:
        return None
    
    # Convert the numpy array image to bytes
    is_success, im_buf_arr = cv2.imencode(".png", uploaded_image)
    byte_im = im_buf_arr.tobytes()

    # Remove the background
    output_bytes = remove(byte_im)

    # Convert the output bytes to an image
    #output_image = image.open(io.BytesIO(output_bytes))
    # Convert the output bytes to an image
    output_image = Image.open(io.BytesIO(output_bytes))


    # Convert the PIL Image back to a numpy array
    image = np.array(output_image)

    # Resize the image while preserving the aspect ratio
    image = resize_image(image, 512)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve blob detection
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area
    params.filterByArea = True
    params.minArea = 20  # Reduced minArea to detect smaller blobs

    # Filter by Circularity (optional)
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(gray_image)
    
    # Number of pips is the number of keypoints
    pip_count = len(keypoints)
    return pip_count

def resize_image(image, target_size):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > h:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

dominoes_folder = "dominoes"

# Test the updated function on the uploaded images
pip_counts = {}
for filename in os.listdir(dominoes_folder):
    # Check if the file is an image (assuming JPG or PNG format)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(dominoes_folder, filename)
        
        # Read the image
        uploaded_image = cv2.imread(img_path)
        
        # Get the pip count
        pip_count = detect_pips(uploaded_image)
        
        # Store the pip count
        pip_counts[filename] = pip_count

print(pip_counts)
