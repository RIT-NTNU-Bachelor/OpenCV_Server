import cv2
import dlib
import numpy as np

def detect_face_mmod(img: np.ndarray, detector: dlib.fhog_object_detector, in_height=300, in_width=0, detect_multiple_faces=False) -> (list | tuple | None) :
    """ Detects faces in an image using the CNN-based dlib MMOD (Max-Margin Object Detection) face detector.

    Parameters:
        img (np.ndarray): The input image in which faces are to be detected. The image should be in a format
                          acceptable by OpenCV, typically a numpy ndarray obtained from `cv2.imread`.
        detector (dlib.fhog_object_detector): The dlib MMOD face detector object, which can be initialized with
                                              `dlib.cnn_face_detection_model_v1` for CNN models.
        in_height (int, optional): The height to which the image will be resized for detection, maintaining the aspect ratio.
                                   Default is 300 pixels.
        in_width (int, optional): The width to which the image will be resized for detection. If set to 0, it will be
                                  calculated based on the aspect ratio of the input image. Defaults to 0.
        detect_multiple_faces (bool, optional): Specifies whether to detect and return bounding boxes for all faces found
                                                in the image or just the most prominent face. Defaults to False.

    Returns:
        If `detect_multiple_faces` is true, returns a list of tuples (x, y, width, height) for each detected face. \
        Else returns a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected. \
        Each tuple represents the top-left corner and dimensions of the bounding box around the detected face.

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

