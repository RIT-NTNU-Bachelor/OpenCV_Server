import cv2
import numpy as np

def detect_face_hog(img: np.ndarray, detector, detect_multiple_faces: bool = False):
    """Detects faces in an image using dlib's HOG-based face detector. 
    
    Read more about the detector here: http://dlib.net/python/index.html#dlib_pybind11.get_frontal_face_detector

    Args:
    - img (np.ndarray): The image in which to detect face. Retrieved by OpenCVs imread function.
    - detector: An instance of the HOG-based detector, pre-trained for face detection. 
                It is initialized with this function: `dlib.get_frontal_face_detector()`
    - detectMultipleFaces (bool, optional): 
            Will return multiple faces if true, else only one. Default is set to true. 

    Returns:
        faces (dlib.rectangle | dlib.rectangles | None)
        - A list of detected face as dlib.rectangles, or
        - A single rectangle for the most prominent face (dlib.rectangle), or
        - None, if no faces are detected.

    """
    # Convert the image to grayscale to simplify the detection process
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection on the grayscale image
    faces = detector(gray_image, 1)

    # Handle the return value based on the detectMultipleFaces flag
    if not detect_multiple_faces:
        return faces[0] if faces else None

    return faces
