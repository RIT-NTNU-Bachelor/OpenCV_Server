import cv2
import dlib
import numpy as np

def detect_face_mmod(img: np.ndarray, detector: dlib.fhog_object_detector, in_height=300, in_width=0, detect_multiple_faces=False):
    """ Detects faces in an image using the CNN-based dlib MMOD (Max-Margin Object Detection) face detector.

    Read more about this detection method in the dlib MMOD documentation: 
    http://dlib.net/python/index.html#dlib.fhog_object_detector

    Parameters: 
    - img (np.ndarray): The input image in which faces are to be detected. It should be in the format acceptable by
                        OpenCV, typically a numpy ndarray obtained from cv2.imread.
    - detector (dlib.fhog_object_detector): The dlib MMOD face detector object, which can be obtained using
                                            dlib.get_frontal_face_detector().
    - inHeight (int, optional): The height of the image for detection. The image will be resized to this height while
                                maintaining the aspect ratio. Defaults to 300.
    - inWidth (int, optional): The width of the image for detection. If set to 0, it will be calculated based on the
                                aspect ratio of the input image. Defaults to 0.
    - detectMultipleFaces (bool, optional): If True, detects and returns bounding boxes for all faces found in the
                                            image. If False, returns the bounding box for the most prominent face.
                                            Defaults to False.

    Returns:
        list or tuple or None( depending on the value of detectMultipleFaces):
        - If True, returns a list of tuples (x, y, width, height) for each detected face.
        - If False, returns a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected.
        - Each tuple contains the coordinates of the top-left corner and the dimensions of the bounding box.

    """

    # Get the dimensions of the input image
    frame_height = img.shape[0]
    frame_width = img.shape[1]
    if not in_width:
        in_width = int((frame_width / frame_height) * in_height)

    # Calculate the scaling factors for height and width
    scale_height = frame_height / in_height
    scale_width = frame_width / in_width

    resized_img = cv2.resize(img, (in_width, in_height))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Perform face detection
    face_rectangles = detector(resized_img, 0)

    # Process the detected faces and calculate the bounding boxes
    faces = []
    for rectangle in face_rectangles:
        x1 = int(rectangle.rect.left() * scale_width)
        y1 = int(rectangle.rect.top() * scale_height)
        x2 = int(rectangle.rect.right() * scale_width)
        y2 = int(rectangle.rect.bottom() * scale_height)
        width = x2 - x1
        height = y2 - y1
        faces.append((x1, y1, width, height))

    # Return detected faces based on the detectMultipleFaces flag
    if detect_multiple_faces:
        return faces  # Return all detected faces
    else:
        return faces[0] if faces else None  # Return the first face or None if no faces are detected

