import cv2
import numpy as np

def detect_face_haar(img: np.ndarray, detector: cv2.CascadeClassifier, detect_multiple_faces: bool = True, scale: float = 1.1, neighbors: int = 10, size: int = 50):
    """Function for detecting faces in an image using a pre-trained Haar Cascade model provided by OpenCV.

    This project uses the "haarcascade_frontalface_default.xml" model, but the function allows for other cascade classifier.
    Default values for scaling, neighbors and size of the window are set. By default the detector will detect multiple faces. Set "detectMultipleFaces" to false for detecting one face. 

    See OpenCV documentation for more information: 
    https://opencv.org/

    Scale, neighbors and size are also explained documentation: 
    https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html 

    Args:
    - img (np.ndarray): 
            The image in which to detect face. Retrieved by OpenCVs imread function.  
    - detector (cv2.CascadeClassifier): 
            An instance of the Haar Cascade detector, pre-trained for face detection.
    - detectMultipleFaces (bool, optional): 
            Will return multiple faces if true, else only one. Default is set to true. 
    - scale (float, optional): 
            Factor by which the image is scaled down to facilitate detection. Default is 1.1, which means scaling it down by 10 percent.
    - neighbors (int, optional): The number of neighbors each candidate rectangle should have to retain it. Defaults to 10.
    - size (int, optional): The minimum size of faces to detect, specified as the side length of the square
                    sliding window used in detection. Defaults to 50 pixels.

    Returns:
        Depending on if detectMultipleFaces is true:
        - A list of tuples (x, y, width, height) for each detected face, or
        - A single tuple (x, y, width, height) for the most prominent face, or
        - None, if no faces are detected.

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
