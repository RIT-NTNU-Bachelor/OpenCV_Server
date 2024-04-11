import cv2
import dlib
import numpy as np

def detect_face_mmod(img: np.ndarray, detector: dlib.fhog_object_detector, inHeight=300, inWidth=0, detectMultipleFaces=False):
    """
    Detects faces in an image using the CNN-based dlib MMOD (Max-Margin Object Detection) face detector.

    Args
    ----------
    (img : np.ndarray): The input image in which faces are to be detected. It should be in the format acceptable by
                        OpenCV, typically a numpy ndarray obtained from cv2.imread.
    (detector : dlib.fhog_object_detector): The dlib MMOD face detector object, which can be obtained using
                                            dlib.get_frontal_face_detector().
    (inHeight : int, optional): The height of the image for detection. The image will be resized to this height while
                                maintaining the aspect ratio. Defaults to 300.
    (inWidth : int, optional): The width of the image for detection. If set to 0, it will be calculated based on the
                                aspect ratio of the input image. Defaults to 0.
    (detectMultipleFaces : bool, optional): If True, detects and returns bounding boxes for all faces found in the
                                            image. If False, returns the bounding box for the most prominent face.
                                            Defaults to False.

    Returns
    -------
    list or tuple or None
        Depending on the value of detectMultipleFaces:
        - If True, returns a list of tuples (x, y, width, height) for each detected face.
        - If False, returns a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected.
        - Each tuple contains the coordinates of the top-left corner and the dimensions of the bounding box.

    References
    ----------
    - dlib MMOD documentation: http://dlib.net/python/index.html#dlib.fhog_object_detector
    """

    # Get the dimensions of the input image
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    # Calculate the scaling factors for height and width
    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    resized_img = cv2.resize(img, (inWidth, inHeight))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Perform face detection
    faceRects = detector(resized_img, 0)

    # Process the detected faces and calculate the bounding boxes
    bboxes = []
    for faceRect in faceRects:
        x1 = int(faceRect.rect.left() * scaleWidth)
        y1 = int(faceRect.rect.top() * scaleHeight)
        x2 = int(faceRect.rect.right() * scaleWidth)
        y2 = int(faceRect.rect.bottom() * scaleHeight)
        width = x2 - x1
        height = y2 - y1
        bboxes.append((x1, y1, width, height))

    # Return detected faces based on the detectMultipleFaces flag
    if detectMultipleFaces:
        return bboxes  # Return all detected faces
    else:
        return bboxes[0] if bboxes else None  # Return the first face or None if no faces are detected

