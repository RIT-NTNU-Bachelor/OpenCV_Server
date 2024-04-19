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
from constants import CVZONE_DETECTOR_MAX_ONE, HAAR_CLASSIFIER, HOG_DETECTOR, DNN_NET, EYE_DISTANCE_INDEX, MMOD_DETECTOR

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

        # Check that the detected 'faces' variable is a instance of np.ndarray
        self.assertIsInstance(faces, np.ndarray, "ERROR: Haar model output should be a np.array")

        # The np.array should have four properties: X, Y, Width, Height
        self.assertEqual(len(faces), 4)

        # Unwrapping the tuple
        x, y, width, height = faces

        # Check that the boundary box is of a certain width and height
        # Using a range to check that the bounding box
        # This is because the box might change slightly position for each iteration 
        self.assertGreater(width, 90)
        self.assertLess(width, 190)
        self.assertGreater(height, 90)
        self.assertLess(height, 190)

        # Checking the position within a set range 
        self.assertGreater(x, 200)
        self.assertLess(x, 220)
        self.assertGreater(y, 190)
        self.assertLess(y, 215)

        # Saving the result to an image
        save_test(self.image,"haar_one_face_output.png", faces)


    # Testing with HOG detector 
    def test_hog_one_face(self):
        faces = detect_face_hog(self.image, HOG_DETECTOR, detectMultipleFaces=False)
        self.assertIsNotNone(faces,"ERROR: HOG model did not find the face in the test with one face")

        # Check that the detected 'faces' variable is an instance of dlib rectangle 
        self.assertIsInstance(faces, dlib.rectangle, "ERROR: HOG model output should be of type dlib rectangle")

        # Check that the boundary box is of a certain width and height
        # Using a range to check that the bounding box
        # This is because the box might change slightly position for each iteration 
        self.assertGreater(faces.width(), 30)
        self.assertLess(faces.width(), 160)
        self.assertGreater(faces.height(), 30)
        self.assertLess(faces.height(), 160)

        # Saving the result to an image
        save_test(self.image,"hog_one_face_output.png", faces)

    # Testing with DNN detector 
    def test_dnn_one_face(self):
        faces = detect_face_dnn(self.image, DNN_NET, detectMultipleFaces=False)
        
        # Check that the detected 'faces' variable is a Rectangle instance 
        self.assertIsInstance(faces, tuple, "ERROR: DNN model output should be a tuple")

        # The tuple should have four properties: X, Y, Width, Height
        self.assertEqual(len(faces), 4)

        # Unwrapping the tuple
        x, y, width, height = faces

        # Check that the boundary box is of a certain width and height
        # Using a range to check that the bounding box
        # This is because the box might change slightly position for each iteration 
        self.assertGreater(width, 120)
        self.assertLess(width, 200)
        self.assertGreater(height, 150)
        self.assertLess(height, 220)

        # Checking the position within a set range 
        self.assertGreater(x, 200)
        self.assertLess(x, 220)
        self.assertGreater(y, 170)
        self.assertLess(y, 215)

        # Saving the result to an image
        save_test(self.image,"dnn_one_face_output.png", faces)

    
    # Testing with MMOD detector 
    def test_mmod_one_face(self):
        faces = detect_face_mmod(self.image, MMOD_DETECTOR, detectMultipleFaces=False)
        
        # Check that the detected faces variable is a tuple instance 
        self.assertIsInstance(faces, tuple, "ERROR: MMOD model output should be a tuple")

        # The tuple should have four properties: X, Y, Width, Height
        self.assertEqual(len(faces), 4)

        # Unwrapping the tuple
        x, y, width, height = faces

        # Check that the boundary box is of a certain width and height
        # Using a range to check that the bounding box
        # This is because the box might change slightly position for each iteration 
        self.assertGreater(width, 120)
        self.assertLess(width, 200)
        self.assertGreater(height, 150)
        self.assertLess(height, 220)

        # Checking the position within a set range 
        self.assertGreater(x, 170)
        self.assertLess(x, 220)
        self.assertGreater(y, 170)
        self.assertLess(y, 215)

        # Saving the result to an image
        save_test(self.image,"mmod_one_face_output.png", faces)

    # Testing with CVZone detector 
    def test_cvzone_one_face(self):
        faces = detect_face_cvzone(self.image, CVZONE_DETECTOR_MAX_ONE, detectMultipleFaces=False)

        # Check that the detected 'faces' variable is a Rectangle instance 
        self.assertIsInstance(faces, list, "ERROR: CVZone model output should be a list")

        # The list should have a list of 468 Landmarks from the mediapipe library.
        self.assertEqual(len(faces), 468)

        # Retrieving the position of the left and right eye
        # Note that the constant has the index of the left and right eye 
        leftEye = faces[EYE_DISTANCE_INDEX['left_eye']]
        rightEye = faces[EYE_DISTANCE_INDEX['right_eye']]

        # Both should be a list with two values. Checking that this is also the case
        self.assertIsInstance(leftEye, list, "ERROR: The left eye should be a tuple")
        self.assertIsInstance(leftEye, list, "ERROR: The right eye should be a tuple")

        # Each tuple has the X and Y position of the landmark 
        # Therefor the length of the tuple should be two for both eye landmarks 
        self.assertEqual(len(leftEye), 2)
        self.assertEqual(len(rightEye), 2)

        # Unwrapping the x and y position for each eye landmark 
        leftEyeX, leftEyeY = leftEye
        rightEyeX, rightEyeY = rightEye

        # Checking that the left eye is left of the right eye
        self.assertLess(leftEyeX, rightEyeX) 

        # In the Lena image, the Y position of the eyes is about the same value.
        # For taking into account that the Y position of the Y can slightly change, we check that they are within a range of a tolerance 
        # Setting the tolerance to 5  
        tolerance = 5 
        
        # Using the tolerance to check that the eyes are is within the excepted range
        self.assertLessEqual(rightEyeY, leftEyeY + tolerance, "ERROR: Right eye is too high compared to the left eye")
        self.assertGreaterEqual(rightEyeY, leftEyeY - tolerance, "ERROR: Right eye is too low compared to the left eye")
        self.assertLessEqual(leftEyeY, rightEyeY + tolerance, "ERROR: Left eye is too high compared to the right eye")
        self.assertGreaterEqual(leftEyeY, rightEyeY - tolerance, "ERROR: Left eye is too low compared to the right eye")


        # Saving the result to an image
        save_test(self.image,"cvzone_one_face_output.png", faces)


# Runs all unit tests within this file 
if __name__ == '__main__':
    unittest.main()