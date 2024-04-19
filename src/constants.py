import cv2
import dlib
from cvzone.FaceMeshModule import FaceMeshDetector 


# ----------------------------------------------------------------------------
# General Configuration Values
# ----------------------------------------------------------------------------
# Average distance between human eyes (in millimeters)
INTEROCULAR_DISTANCE = 63

# Focal length of the camera in use (in some unit)
FOCAL_LENGTH = 655

# ----------------------------------------------------------------------------
# DNN Model Configuration
# ----------------------------------------------------------------------------
# Path configurations for DNN models
DNN_CAFFE_MODEL_PATH = "./src/models/trained_models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
DNN_CONFIG_PATH = "./src/models/trained_models/deploy.prototxt"

# Deep Neural Network (DNN) model loaded using OpenCV
DNN_NET = cv2.dnn.readNetFromCaffe(DNN_CONFIG_PATH, DNN_CAFFE_MODEL_PATH)

# ----------------------------------------------------------------------------
# Haar Cascade Detector Configuration
# ----------------------------------------------------------------------------
# Haar Cascade classifier for face detection
HAAR_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------------------------------------------------------
# HOG Detector Configuration
# ----------------------------------------------------------------------------
# Histogram of Oriented Gradients (HOG) face detector from Dlib
HOG_DETECTOR = dlib.get_frontal_face_detector()

# ----------------------------------------------------------------------------
# MMOD Detector Configuration
# ----------------------------------------------------------------------------
# Max-Margin Object Detection (MMOD) model from Dlib
MMOD_DETECTOR = dlib.cnn_face_detection_model_v1("./src/models/trained_models/mmod_human_face_detector.dat")

# ----------------------------------------------------------------------------
# CVZone Face Mesh Detector Configuration
# ----------------------------------------------------------------------------
# Face mesh detector with default settings from CVZone
CVZONE_DETECTOR = FaceMeshDetector()

# Face mesh detector configured to detect a maximum of one face
CVZONE_DETECTOR_MAX_ONE = FaceMeshDetector(maxFaces=1)

# Constants for CVZone face mesh indices for eye locations
EYE_DISTANCE_INDEX = {
    'left_eye': 145,
    'right_eye': 374
}