# Importing the unit tests package 
import unittest

# Importing the OpenCV library for reading an image
import cv2

# Import dlib library
import dlib

# Import numpy for checking the instance of Haar output
import numpy as np

# Setup the correct path 
from tests.test_utils import set_project_path_for_tests, save_test
set_project_path_for_tests()

# Import the source code for the all of the models 
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog
from models.code.dnn import detect_face_dnn
from models.code.mmod import detect_face_mmod
from models.code.cvzone import detect_face_cvzone

# If this gives an error, the requirements.txt has not been correctly installed
from constants import CVZONE_DETECTOR, HAAR_CLASSIFIER, HOG_DETECTOR, DNN_NET, MMOD_DETECTOR

# Path to the image with two faces
# All models should be able to detect both faces in the image 
# The image is taken from a podcast episode cover produced by the company Goop.
# Here is the image source: https://goop.com/goopfellas-podcast/stan-tatkin-what-keeps-two-people-together/
path_to_face_image = "data/test_data/unit_test/TwoFaces.jpg"


# The test class
class TestModelsWithTwoFaces(unittest.TestCase):

    # Setting up the test unit before each test. 
    # The setup is called once when the test class is set up
    # The same image is used for all test cases 
    @classmethod
    def setUp(self):
        self.image = cv2.imread(path_to_face_image)

    # Test that checks that the image has been correctly setup 
    def test_image_loaded(self):
        self.assertIsNotNone(self.image, "ERROR: Image should be correctly setup for the tests with two faces")

    # Test with Haar face detection model 
    def test_haar_two_faces(self):
        faces = detect_face_haar(self.image, HAAR_CLASSIFIER, detect_multiple_faces=True)

        # Check that the detected 'faces' variable is a instance of np.ndarray
        self.assertIsInstance(faces, np.ndarray, "ERROR: Haar model output should be a np.array")

        # The np.array should have two faces
        self.assertEqual(len(faces), 2)

        # For each face check that they are a numpy array
        # Also if the array has the four expected values: X, Y, Width and Height 
        for face in faces: 
            self.assertIsInstance(face, np.ndarray, "ERROR: A face was not a numpy array")
            self.assertEqual(len(face), 4, "ERROR: A face did not have the four properties expected in the numpy array")

        # Saving the result to an image
        save_test(self.image,"haar_two_faces_output.png", faces)

    # Testing with HOG detector 
    def test_hog_two_faces(self):
        faces = detect_face_hog(self.image, HOG_DETECTOR, detect_multiple_faces=True)
        self.assertIsNotNone(faces,"ERROR: HOG model did not find the face in the test with one face")

        # Check that the detected 'faces' variable is an instance of dlib rectangles
        # This is a list of rectangles 
        self.assertIsInstance(faces, dlib.rectangles, "ERROR: HOG model output should be of type dlib rectangles with the two faces")

        # There should be detected two faces in this list of rectangles
        self.assertEqual(len(faces), 2)

        # For each face check that they are a dlib rectangle 
        # Also if the rectangle object has a value for the width and height properties of the boundary box
        for face in faces: 
            self.assertIsInstance(face, dlib.rectangle, "ERROR: A face was not a rectangle")
            self.assertIsNotNone(face.width())
            self.assertIsNotNone(face.height())

        # Saving the result to an image
        save_test(self.image,"hog_two_faces_output.png", faces)


    # Testing with DNN detector 
    def test_dnn_two_faces(self):
        faces = detect_face_dnn(self.image, DNN_NET, detect_multiple_faces=True)
        
        # Check that the detected 'faces' variable is a list instance 
        self.assertIsInstance(faces, list, "ERROR: DNN model output should be a list with the two faces")

        # There should be detected two faces in this list
        self.assertEqual(len(faces), 2)

        # For each face check that they are a tuple 
        # Also if the tuple has the four expected values: X, Y, Width and Height 
        for face in faces: 
            self.assertIsInstance(face, tuple, "ERROR: A face was not a tuple")
            self.assertEqual(len(face), 4, "ERROR: A face did not have the four properties expected in the rectangle")

        # Saving the result to an image
        save_test(self.image,"dnn_two_faces_output.png", faces)


    # Testing with MMOD detector 
    def test_mmod_two_faces(self):
        faces = detect_face_mmod(self.image, MMOD_DETECTOR, detect_multiple_faces=True)
        
        # Check that the detected faces variable is a list instance 
        self.assertIsInstance(faces, list, "ERROR: DNN model output should be a list with the two faces")

        # There should be detected two faces in this list
        self.assertEqual(len(faces), 2)

        # For each face check that they are a tuple 
        # Also if the tuple has the four expected values: X, Y, Width and Height 
        for face in faces: 
            self.assertIsInstance(face, tuple, "ERROR: A face was not a tuple")
            self.assertEqual(len(face), 4, "ERROR: A face did not have the four properties expected in the rectangle")

        # Saving the result to an image
        save_test(self.image,"mmod_two_faces_output.png", faces)

    # Testing with CVZone detector 
    def test_cvzone_two_faces(self):
        faces = detect_face_cvzone(self.image, CVZONE_DETECTOR, detect_multiple_faces=True)

        # Check that the detected 'faces' variable is a Rectangle instance 
        self.assertIsInstance(faces, list, "ERROR: CVZone model output should be a list")

        # There should be detected two faces in this list
        self.assertEqual(len(faces), 2)

        # For each face check that they are a list of landmarks
        # Also if the list has all of the 468 landmarks 
        for face in faces: 
            self.assertIsInstance(face, list, "ERROR: A face was not a list of the expected landmarks")
            self.assertEqual(len(face), 468, "ERROR: A face did not have the 468 landmarks expected")

        # Saving the result to an image
        save_test(self.image,"cvzone_two_faces_output.png", faces)

# Runs all unit tests within this file 
if __name__ == '__main__':
    unittest.main()