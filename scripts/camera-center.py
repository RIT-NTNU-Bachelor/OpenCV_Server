# Import OpenCV image
import cv2 

# Create the image instance
img = None; 

# Initialize the camera
capture = cv2.VideoCapture(0)  

# Check if the camera can be opened 
if not capture.isOpened():
    # Logging errors in console
    print("Error: Could not open camera.\n")
    print("Please confirm that the camera is connected")
    
    # Stop the script
    exit(1)
else:
    # Capture one frame
    isCaptured, frame = capture.read()

    # Check if the frame was captured successfully
    if isCaptured:
        img = frame
    else:
        print("Error: Could not capture an image.")
        exit(1)

# Release the camera
capture.release()

# Image should be set. Image should not be none here
assert img.any() != None, "Error: Could not retrieve captured image"
print(f"Width = {img.shape[1]}, Height = {img.shape[0]}\n")

# Printing what the center of the configuration value 
print("Values to be set in Unreal Engine Client:")
print(f"CX = {img.shape[1] / 2}, CY = {img.shape[0] / 2}" )