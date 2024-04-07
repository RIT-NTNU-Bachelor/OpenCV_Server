import cv2
import numpy as np

def detect_face_haar(img: np.ndarray, detector: cv2.CascadeClassifier, detectMultipleFaces: bool = True, scale: float = 1.1, neighbors: int = 10, size: int = 50):
    """
    Function for detecting faces in an image using a pre-trained Haar Cascade model provided by OpenCV.
    This project uses the "haarcascade_frontalface_default.xml" model, but the function allows for other cascade classifier.
    Default values for scaling, neighbors and size of the window are set. 

    By default the detector will detect multiple faces. Set "detectMultipleFaces" to false for detecting one face. 

    See OpenCV documentation for more information: 
    https://opencv.org/

    Scale, neighbors and size are also explained documentation: 
    https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html 

    Args:
        img (np.ndarray): 
            The image in which to detect face. Retrieved by OpenCVs imread function.  
        detector (cv2.CascadeClassifier): 
            An instance of the Haar Cascade detector, pre-trained for face detection.
        detectMultipleFaces (bool): 
            Will return multiple faces if true, else only one. Default is set to true. 
        scale (float): 
            Factor by which the image is scaled down to facilitate detection. Default is 1.1, which means scaling it down by 10 percent.
        neighbors (int): The number of neighbors each candidate rectangle should have to retain it. Defaults to 10.
        size (int): The minimum size of faces to detect, specified as the side length of the square
                    sliding window used in detection. Defaults to 50 pixels.

    Returns:
        If detectMultipleFaces is True, returns a numpy ndarray of rectangles, where each rectangle,
        represented as (x, y, width, height), corresponds to a detected face. If False, returns a single
        tuple (x, y, width, height) for the most prominent face, or None if no face is detected.
    """
    # Convert the image to a grayscale image to simplify detection
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = detector.detectMultiScale(
        gray_image, scaleFactor=scale, minNeighbors=neighbors, minSize=(size, size)
    )

    # Return either multiple faces or the most prominent one
    if detectMultipleFaces:
        return faces
    else:
        return faces[0] if len(faces) > 0 else None
