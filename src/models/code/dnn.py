import cv2

def detect_face_dnn(img, net, framework="caffe", conf_threshold=0.7, detectMultipleFaces=False):
    """
    Detect faces in an image using a deep neural network (DNN).

    Parameters:
    - img: The input image.
    - net: The pre-trained DNN model for face detection.
    - framework: The framework used for the DNN model ('caffe' or 'tensorflow').
    - conf_threshold: The confidence threshold for detecting faces.
    - detect_multiple_faces: Boolean flag to detect multiple faces or just the first one.

    Returns:
    - A list of bounding boxes for detected faces or a single bounding box if detectMultipleFaces is False.
    """
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    if framework == "caffe":
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    else:
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            width = x2 - x1
            height = y2 - y1
            bboxes.append((x1, y1, width, height))

    if detectMultipleFaces == True:
        return bboxes  # Return all detected faces
    else:
        return bboxes[0] if bboxes else None # Return the first face or None if no faces are detected