import cv2

def detect_face_haar(img,detectMultipleFaces=False, scale=1.1, neighbors=10, size=50):
    """Detect a face in an image using a pre-trained Haar Cascasde model. 

    The model has been trained by OpenCV.
    See: https://opencv.org/

    Args:
        img (numpy.ndarray): 
            Image read from the cv2.imgread function. Ut is a numpy
        detectMultipleFaces (boolean): 
            Toggle for returning more than one face detected. Default is false. 
        scale (float, optional): 
            For scaling down the input image, before trying to detect a face. Makes it easier to detect a face with smaller scale. Defaults to 1.1.
        neighbors (int, optional): 
            Amount of neighbour rectangles needed for a face to be set as detected. Defaults to 10.
        size (int, optional): 
            Size of the sliding window that checks for any facial features. Should match the face size in the image, that should be detected. Defaults to 50.

    Returns:
        Rect: Datatype of a rectangle, that overlays the position of the detected face. It has four attributes of intrests: x-position, y-position, 
    """

    # Turing the image into a grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Printing the gray scale image
    # print(f"Gray-Scale Image dimension: ({gray_image.shape})")

    # Loading the classifier from a pretrained dataset
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Performing the face detection
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=scale, minNeighbors=neighbors, minSize=(size,size)
    )

    # Return amount of 
    if detectMultipleFaces == True:
        return faces
    return faces[0]