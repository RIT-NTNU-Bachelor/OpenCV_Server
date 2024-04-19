import cv2
import numpy as np

def detect_face_dnn(img: np.ndarray, net: cv2.dnn_Net, framework: str = "caffe", conf_threshold: float = 0.7, detect_multiple_faces: bool = False) -> (list | tuple | None):
    """Function that detects faces in an image using a Deep Neural Network (DNN) model. The function supports models trained
    with either the Caffe or TensorFlow framework.

    Parameters:
        img (np.ndarray): The input image in which faces are to be detected. The image should be in a format
                          acceptable by OpenCV, typically a numpy ndarray obtained from `cv2.imread`.
        net (cv2.dnn_Net): The pre-trained DNN model loaded using `cv2.dnn.readNet` for face detection. 
                           For more information on loading models, refer to the [OpenCV documentation on dnn_Net](https://docs.opencv.org/master/db/d30/classcv_1_1dnn_1_1Net.html).
        framework (str, optional): Specifies the framework of the pre-trained model. Can be 'caffe' or
                                   'tensorflow'. Defaults to 'caffe'.
        conf_threshold (float, optional): The minimum confidence threshold for a detection to be considered valid. 
                                          Ranges between 0 and 1, with a higher threshold reducing false positives. Defaults to 0.7.
        detect_multiple_faces (bool, optional): If True, detects and returns bounding boxes for all detected faces.
                                                If False, returns a bounding box for the most prominent face or None if no faces are detected.

    Returns:       
        If True, returns a list of tuples (x, y, width, height) for each detected face. If False, returns a single tuple (x, y, width, height) for the most prominent face, or None if no faces are detected. Each tuple contains the coordinates of the top-left corner and the dimensions of the bounding box.

    Note:
        This function requires that the appropriate DNN model files are accessible and properly configured before use. 
        The [OpenCV Server](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server) code has this setup by default. 
    """

    # Get the dimensions of the input image
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]

    # Prepare the blob from the image (input to the DNN)
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], swapRB=(framework != "caffe"), crop=False)

    # Set the prepared blob as the input to the network
    net.setInput(blob)

    # Forward the value through the network 
    # The output of the DNN is the detections made in the image
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
    if detect_multiple_faces:
        return faces  # Return all detected faces
    else:
        return faces[0] if faces else None  # Return the first detected face or None