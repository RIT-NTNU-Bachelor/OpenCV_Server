# Importing the unit tests package 
import unittest

# For testing we need to move within the src folder to find the module
import os
import sys

# Adjust the path to include the directory where modules module is located
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../src')  
sys.path.insert(0, parent_dir)

# Import the source code for the all of the 
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog
from models.code.dnn import detect_face_dnn
from models.code.cvzone import detect_face_cvzone

# Path to the image with one clear face
# All models should be able to detect the face in the image 
# The image is standardized image in the field of Computer Vision, also known as Lenna. 
# Read more about the image here: https://en.wikipedia.org/wiki/Lenna
path_to_face_image = "../data/test_data/unit_test/Lenna.png"


class TestModelsWithOneFace(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1,1)







# Runs all unit tests within this file 
if __name__ == '__main__':
    unittest.main()