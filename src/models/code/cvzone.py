import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

def detect_face_cvzone(img: np.ndarray, detector: FaceMeshDetector, detectMultipleFaces=True):
    """
    Function that detects faces in an image using the CVZone library. 

    Link to the library: https://github.com/cvzone/cvzone 

    Args:
        img (np.ndarray): 
            The image in which to detect face. Retrieved by OpenCVs imread function.  
        detector (FaceMeshDetector): 
            An instance of the Face Mesh detector, pretrained from the CVZone library
        detectMultipleFaces (bool): 
            Will return multiple faces if true, else only one. Default is set to true. 

    Returns:
        Depending on if detectMultipleFaces is true:
        - A list of tuples (x, y, width, height) for each detected face, or
        - A single tuple (x, y, width, height) for the most prominent face, or
        - None, if no faces are detected.

    Each tuple contains the coordinates of the top left corner and the dimensions of the bounding box.
    """
    _, faces = detector.findFaceMesh(img, draw=False)

    if detectMultipleFaces == True:
        return faces
    return faces[0] if faces else None