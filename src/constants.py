import cv2
import dlib
from cvzone.FaceMeshModule import FaceMeshDetector # Import for CVZone


# Constants for the DNN model
DNN_CAFFE_MODEL_PATH = "./src/models/trained_models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
DNN_CONFIG_PATH = "./src/models/trained_models/deploy.prototxt"
DNN_NET = cv2.dnn.readNetFromCaffe(DNN_CONFIG_PATH, DNN_CAFFE_MODEL_PATH)

# Constants for Haar
HAAR_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Constants for Hog
HOG_DETECTOR = dlib.get_frontal_face_detector()

# Constants for CVZone
CVZONE_DETECTOR = FaceMeshDetector()
CVZONE_DETECTOR_MAX_ONE = FaceMeshDetector(maxFaces=1)