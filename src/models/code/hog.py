import cv2
import dlib

def detect_face_hog(img,detectMultipleFaces=False):
    # Turing the image into a grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the HOG detector from dlib
    hog_face_detector = dlib.get_frontal_face_detector()

    # Detect faces from the grayscale image
    faces = hog_face_detector(gray_image, 1)

    # Check if only one face is needed
    if detectMultipleFaces == False:
        return faces[0]

    return faces