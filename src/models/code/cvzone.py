import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

def detect_face_cvzone(img: np.ndarray, detector: FaceMeshDetector, detect_multiple_faces=True) -> (list | tuple | None):
    """ Function that detects faces in an image using the CVZone library. 

    Parameters:
        img (np.ndarray): The image in which to detect face. Retrieved by OpenCVs imread function.  
        detector (FaceMeshDetector): An instance of the Face Mesh detector, pretrained from the CVZone library
        detect_multiple_faces (bool, optional): Will return multiple faces if true, else only one. Default is set to true. 

    Returns:
        If `detect_multiple_faces` is true, returns a list of tuples (x, y, width, height) for each detected face. \
        Else returns a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected. \
        Each tuple represents the top-left corner and dimensions of the bounding box around the detected face.

    Each tuple contains the coordinates of the top left corner and the dimensions of the bounding box.
    """
    _, faces = detector.findFaceMesh(img, draw=False)

    if detect_multiple_faces == True:
        return faces
    return faces[0] if faces else None