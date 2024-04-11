import cv2
import numpy as np

def detect_face_dnn(img: np.ndarray, net: cv2.dnn_Net, framework: str = "caffe", conf_threshold: float = 0.7, detectMultipleFaces: bool = False):
    """
    Function that detects faces in an image using a Deep Neural Network (DNN) model. The function supports models trained
    with either the Caffe or TensorFlow framework.

    Args
    ----------
    img (np.ndarray): The input image in which faces are to be detected. It should be in the format
                          acceptable by OpenCV, typically a numpy ndarray obtained from cv2.imread.
    net (cv2.dnn_Net): The pre-trained DNN model loaded using cv2.dnn.readNet for face detection. For more information
                        on loading models, refer to: https://docs.opencv.org/master/db/d30/classcv_1_1dnn_1_1Net.html
    framework (str, optional): Specifies the framework of the pre-trained model. Accepts either 'caffe' or
                         'tensorflow'. Defaults to 'caffe'.
    conf_threshold (float, optional): The minimum confidence level for a detection to be considered a face.
                                Ranges between 0 and 1, with a higher threshold resulting in fewer
                                detections but with increased reliability. Defaults to 0.7.
    detectMultipleFaces (bool, optional): If True, detects and returns bounding boxes for all faces found in
                                    the image. If False, returns the bounding box for the


    Returns
    -------
    list or tuple or None
        Depending on the value of detectMultipleFaces:
        - If True, returns a list of tuples (x, y, width, height) for each detected face.
        - If False, returns a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected.
        - Each tuple contains the coordinates of the top-left corner and the dimensions of the bounding box.

     References
    ----------
    - OpenCV DNN documentation: https://docs.opencv.org/master/d6/d0f/group__dnn.html
    - Caffe models: https://caffe.berkeleyvision.org/
    - TensorFlow models: https://www.tensorflow.org/
    """

    # Get the dimensions of the input image
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]

    # Prepare the blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], swapRB=(framework != "caffe"), crop=False)

    # Set the prepared blob as the input to the network
    net.setInput(blob)

    # Perform inference and obtain the detections
    detections = net.forward()

    # Process detections and extract bounding boxes
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            width = x2 - x1
            height = y2 - y1
            faces.append((x1, y1, width, height))

    # Return detected faces based on the detectMultipleFaces flag
    if detectMultipleFaces:
        return faces  # Return all detected faces
    else:
        return faces[0] if faces else None  # Return the first detected face or None