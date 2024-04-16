# Importing the unit tests package 
import unittest

# Importing the OpenCV library for reading an image
import cv2

# Import dlib library
import dlib

# Setup the correct path 
from tests.test_utils import set_project_path_for_tests
set_project_path_for_tests()

# Import the source code for the all of the models 
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog
from models.code.dnn import detect_face_dnn
from models.code.cvzone import detect_face_cvzone

# If this gives an error, the requirements.txt has not been correctly installed
from constants.model_constants import CVZONE_DETECTOR_MAX_ONE, HAAR_CLASSIFIER, HOG_DETECTOR, DNN_NET

# Path to the image with one clear face
# All models should be able to detect the face in the image 
# The image is standardized image in the field of Computer Vision, also known as Lenna. 
# Read more about the image here: https://en.wikipedia.org/wiki/Lenna
path_to_face_image = "data/test_data/unit_test/Lenna.png"


# The test class 
class TestModelsWithOneFace(unittest.TestCase):

    # Setting up the test unit before each test. 
    # The setup is called once when the test class is set up
    # The same image is used for all test cases 
    @classmethod
    def setUp(self):
        self.image = cv2.imread(path_to_face_image)

    # Test that checks that the image has been correctly setup 
    def test_image_loaded(self):
        self.assertIsNotNone(self.image, "ERROR: Image should be correctly setup for the tests with one face")

    # Test with Haar face detection model 
    def test_haar_one_face(self):
        faces = detect_face_haar(self.image, HAAR_CLASSIFIER, detectMultipleFaces=False)
        self.assertEqual(len(faces),1, "ERROR: Har model did not find the face in the test with one face")

    # Testing with HOG detector 
    def test_hog_one_face(self):
        faces = detect_face_hog(self.image, HOG_DETECTOR, detectMultipleFaces=False)
        self.assertIsNotNone(faces,"ERROR: HOG model did not find the face in the test with one face")

        # Check that the detected 'faces' variable is a rectangle of length 4 (x, y, width, height)
        self.assertIsInstance(faces, dlib.rectangle, "ERROR: HOG model output should be a dlib rectangle")

        # Check that the boundary box is of a certain width and height
        # Using a range to check that the bounding box
        # This is because the box might change slightly position for each iteration 
        self.assertGreater(faces.width(), 30)
        self.assertLess(faces.width(), 160)
        self.assertGreater(faces.height(), 30)
        self.assertLess(faces.height(), 160)

    # Testing with DNN detector 
    def test_dnn_one_face(self):
        faces = detect_face_dnn(self.image, DNN_NET, detectMultipleFaces=False)
        self.assertEqual(len(faces),1)

    # Testing with CVZone detector 
    def test_cvzone_one_face(self):
        faces = detect_face_cvzone(self.image, CVZONE_DETECTOR_MAX_ONE, detectMultipleFaces=False)
        self.assertEqual(len(faces),1)



# Runs all unit tests within this file 
if __name__ == '__main__':
    unittest.main()