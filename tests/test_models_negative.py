# Importing the unit tests package 
import unittest

# Importing the OpenCV library for reading an image
import cv2

# Setup the correct path 
from tests.test_utils import set_project_path_for_tests
set_project_path_for_tests()

# Import the source code for the all of the models 
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog
from models.code.dnn import detect_face_dnn
from models.code.mmod import detect_face_mmod
from models.code.cvzone import detect_face_cvzone

# If this gives an error, the requirements.txt has not been correctly installed
from constants import CVZONE_DETECTOR_MAX_ONE, HAAR_CLASSIFIER, HOG_DETECTOR, DNN_NET, MMOD_DETECTOR


# Path to the image without a face
# Image taken from the USC-SIPI Image Database: https://sipi.usc.edu/database/
path_to_peppers_image = "data/unit_test/Peppers.png"


# The test class with negative tests 
# The negative tests will check that there are no faces detected in an image without a any faces
class TestModelsNegative(unittest.TestCase):

    # Setting up the test unit before each test. 
    # The setup is called once when the test class is set up
    # The same image is used for all test cases 
    @classmethod
    def setUp(self):
        self.image = cv2.imread(path_to_peppers_image)

    # Test that checks that the image has been correctly setup 
    def test_image_loaded(self):
        self.assertIsNotNone(self.image, "ERROR: Image should be correctly setup for the tests with one face")

    # Test with Haar face detection model 
    def test_haar_negative(self):
        faces = detect_face_haar(self.image, HAAR_CLASSIFIER, detect_multiple_faces=False)

        # Assert that it is none => no face detected
        self.assertIsNone(faces)

    # Testing with HOG detector 
    def test_hog_negative(self):
        faces = detect_face_hog(self.image, HOG_DETECTOR, detect_multiple_faces=False)

        # Assert that it is none => no face detected
        self.assertIsNone(faces)


    # Testing with DNN detector 
    def test_dnn_negative(self):
        faces = detect_face_dnn(self.image, DNN_NET, detect_multiple_faces=False)
        
        # Assert that it is none => no face detected
        self.assertIsNone(faces)

    # Testing with DNN detector 
    def test_mmod_negative(self):
        faces = detect_face_mmod(self.image, MMOD_DETECTOR, detect_multiple_faces=False)
        
        # Assert that it is none => no face detected
        self.assertIsNone(faces)

    # Testing with CVZone detector 
    def test_cvzone_negative(self):
        faces = detect_face_cvzone(self.image, CVZONE_DETECTOR_MAX_ONE, detect_multiple_faces=False)

        # Assert that it is none => no face detected
        self.assertIsNone(faces)

# Runs all unit tests within this file 
if __name__ == '__main__':
    unittest.main()