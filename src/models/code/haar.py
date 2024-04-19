import cv2
import numpy as np

def detect_face_haar(img: np.ndarray, detector: cv2.CascadeClassifier, detect_multiple_faces: bool = True, scale: float = 1.1, neighbors: int = 10, size: int = 50) -> (list | tuple | None):
    """Function for detecting faces in an image using a pre-trained Haar Cascade model provided by OpenCV.

    This project uses the "haarcascade_frontalface_default.xml" model, but the function allows for other cascade classifier.
    Default values for scaling, neighbors and size of the window are set. By default the detector will detect multiple faces. Set `detect_multiple_faces` to false for detecting one face. 
 
    Parameters:
        img (np.ndarray): The image in which faces are to be detected, typically obtained from `cv2.imread`.
        detector (cv2.CascadeClassifier): An instance of Haar Cascade detector, pre-trained for face detection.
        detect_multiple_faces (bool, optional): Controls whether to detect multiple faces or just the most prominent one.
                                                Defaults to True, detecting multiple faces.
        scale (float, optional): The factor by which the image is scaled down to facilitate detection. Scaling down the image
                                 can lead to faster detection with less precision. Defaults to 1.1 (10% reduction).
        neighbors (int, optional): The number of neighbors each candidate rectangle should have to retain it. 
                                   A higher number gives fewer detections but with higher quality. Defaults to 10.
        size (int, optional): The minimum size of faces to detect, specified as the side length of the square
                              sliding window used in detection. Defaults to 50 pixels.

    Returns:
        If `detect_multiple_faces` is true, returns a list of tuples (x, y, width, height) for each detected face. \
        Else return a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected. \
        Each tuple represents the top-left corner and dimensions of the bounding box around the detected face. \
        Each tuple contains the coordinates of the top left corner and the dimensions of the bounding box.
    """
    # Convert the image to a grayscale image to simplify detection
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = detector.detectMultiScale(
        gray_image, scaleFactor=scale, minNeighbors=neighbors, minSize=(size, size)
    )

    # Return either multiple faces or the most prominent one
    if detect_multiple_faces:
        return faces
    else:
        return faces[0] if len(faces) > 0 else None
