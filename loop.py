import cv2
import os

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

# Path to the folder containing domino images
dominoes_folder = "dominoes"

# Loop through each file in the folder
for filename in os.listdir(dominoes_folder):
    # Check if the file is an image (assuming JPG format, you can add more types)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(dominoes_folder, filename)
        
        # Read the image
        uploaded_image = cv2.imread(file_path)
        
        # Get the pip count
        pip_count = detect_pips(uploaded_image)
        
        # Print the pip count
        print(f'Total pip count for {filename} is: {pip_count}')
