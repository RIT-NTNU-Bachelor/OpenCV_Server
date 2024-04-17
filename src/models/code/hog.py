import cv2
import dlib
import numpy as np

def detect_face_hog(img: np.ndarray, detector, detectMultipleFaces: bool = False):
    """
    Detects faces in an image using dlib's HOG-based face detector. 
    
    Read more about the detector here: http://dlib.net/python/index.html#dlib_pybind11.get_frontal_face_detector

    Args:
        img (np.ndarray): 
            The image in which to detect face. Retrieved by OpenCVs imread function.

        detector: 
            An instance of the HOG-based detector, pre-trained for face detection.  
            It is initialized with this function: `dlib.get_frontal_face_detector()`

        detectMultipleFaces (bool): 
            Will return multiple faces if true, else only one. Default is set to true. 

    Returns:
        faces (dlib.rectangle | dlib.rectangle[])
            Depending on if detectMultipleFaces is true:
            - A list of tuples (x, y, width, height) for each detected face, or
            - A single tuple (x, y, width, height) for the most prominent face, or
            - None, if no faces are detected.

            Each tuple contains the coordinates of the top left corner and the dimensions of the bounding box.
    """
    # Convert the image to grayscale to simplify the detection process
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the HOG face detector from dlib
    hog_face_detector = dlib.get_frontal_face_detector()

    # Perform face detection on the grayscale image
    faces = hog_face_detector(gray_image, 1)

    # Handle the return value based on the detectMultipleFaces flag
    if not detectMultipleFaces:
        return faces[0] if faces else None

    return faces
