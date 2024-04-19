import cv2
import numpy as np
import dlib

def detect_face_hog(img: np.ndarray, detector, detect_multiple_faces: bool = False) -> (dlib.rectangle | dlib.rectangles | None):
    """Detects faces in an image using dlib's HOG-based face detector. 
    

    Parameters:
        img (np.ndarray): The image in which faces are to be detected, typically obtained from `cv2.imread`.
        detector: An instance of dlib's HOG-based face detector, typically initialized using `dlib.get_frontal_face_detector()`.
        detect_multiple_faces (bool, optional): Specifies whether the function should return detections for all faces found 
                                                (True) or just the most prominent face (False). Defaults to False.

    Returns: 
        If `detect_multiple_faces` is True, returns a list of `dlib.rectangles` each indicating a detected face. \
        If False, returns a single `dlib.rectangle` for the most prominent face, or None if no faces are detected. \
        Each `dlib.rectangle` object represents the bounding box around a detected face with attributes allowing access to the bounding coordinates. \


    """
    # Convert the image to grayscale to simplify the detection process
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection on the grayscale image
    faces = detector(gray_image, 1)

    # Handle the return value based on the detectMultipleFaces flag
    if not detect_multiple_faces:
        return faces[0] if faces else None

    return faces
