import cv2

# TODO: Delete or fix 

def detect_face_mmod(img, detector, inHeight=300, inWidth=0, detectMultipleFaces=False):
    """
    Detect faces in an image using the dlib MMOD detector.

    Parameters:
    - img: The input image.
    - detector: The dlib MMOD face detector.
    - inHeight: The height of the image for detection.
    - inWidth: The width of the image for detection. If 0, it will be calculated based on the aspect ratio of the input image.
    - detectMultipleFaces: Boolean flag to indicate whether to detect multiple faces or just the first one.

    Returns:
    - A list of bounding boxes for each detected face or a single bounding box if detectMultipleFaces is False.
    """
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    resized_img = cv2.resize(img, (inWidth, inHeight))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    faceRects = detector(resized_img, 0)

    bboxes = []
    for faceRect in faceRects:
        x1 = int(faceRect.rect.left() * scaleWidth)
        y1 = int(faceRect.rect.top() * scaleHeight)
        x2 = int(faceRect.rect.right() * scaleWidth)
        y2 = int(faceRect.rect.bottom() * scaleHeight)
        width = x2 - x1
        height = y2 - y1
        bboxes.append((x1, y1, width, height))

    if detectMultipleFaces == True:
        return bboxes  # Return all detected faces
    else:
        return bboxes[0] if bboxes else None  # Return the first face or None if no faces are detected

