# Import OpenCV for reading image
import cv2

# Importing the functions for the face detection models 
from models.code.dnn import detect_face_dnn
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog
from models.code.mmod import detect_face_mmod
from models.code.cvzone import detect_face_cvzone

# Import the detectors from src/constants.py
from constants.model_constants import DNN_NET, CVZONE_DETECTOR_MAX_ONE, HAAR_CLASSIFIER, HOG_DETECTOR, MMOD_DETECTOR

# Opening a sample image
image_path = "path/to/your/image.jpg"
img = cv2.imread(image_path)

# Example of using a method to detect face models 
faces_dnn = detect_face_dnn(img, DNN_NET)
faces_haar = detect_face_haar(img, HAAR_CLASSIFIER)
faces_hog = detect_face_hog(img, HOG_DETECTOR)
faces_cvzone = detect_face_cvzone(img, CVZONE_DETECTOR_MAX_ONE)
faces_mmod = detect_face_mmod(img,MMOD_DETECTOR)

# Print the amount of faces found within the image
print(f"DNN Detected Faces: {faces_dnn}")
print(f"Haar Detected Faces: {faces_haar}")
print(f"HOG Detected Faces: {faces_hog}")
print(f"CVZone Detected Faces: {faces_cvzone}")
print(f"MMOD Detected Faces: {faces_mmod}")


