# Importing the functions for the face detection models 
from models.code.dnn import detect_face_dnn
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog

# Import OpenCV for reading image
import cv2

# Import the detectors from src/constants.py
from constants import DNN_NET


# Main function
def main():
    # Opening a sample image
    image_path = "path/to/your/image.jpg"
    img = cv2.imread(image_path)


    # Example of using a method to detect face models 
    faces_dnn = detect_face_dnn(img, DNN_NET)
    faces_haar = detect_face_haar(img, )
    faces_hog = detect_face_hog(img)
   
    # Display results
    print(f"DNN Detected Faces: {faces_dnn}")
    print(f"Haar Detected Faces: {faces_haar}")
    print(f"HOG Detected Faces: {faces_hog}")


# Run the main function when the file is runned 
if __name__ == "__main__":
    main()
