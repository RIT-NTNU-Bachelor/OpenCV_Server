# Importing the unit tests package 
import unittest

# Setup the correct path 
from tests.test_utils import set_project_path_for_tests
set_project_path_for_tests()

# If this gives an error, the requirements.txt has not been correctly installed
from constants import CVZONE_DETECTOR, CVZONE_DETECTOR_MAX_ONE, HAAR_CLASSIFIER, HOG_DETECTOR, DNN_NET, MMOD_DETECTOR

class TestLoadModels(unittest.TestCase):
    
    # Testing the Haar model correctly loaded. 
    def test_load_haar(self):
        self.assertIsNotNone(HAAR_CLASSIFIER, "Haar classifier should not be none")

    # Testing that the hog model is correctly loaded
    def test_load_hog(self):
        self.assertIsNotNone(HOG_DETECTOR, "HOG detector from dlib library should not be none")

    # Testing that the DNN model is imported correctly 
    def test_load_dnn(self):
        self.assertIsNotNone(DNN_NET, "DNN model could not be loaded")

    # Testing that the MMOD model is imported correctly 
    def test_load_mmod(self):
        self.assertIsNotNone(MMOD_DETECTOR, "MMOD model could not be loaded")

    # Testing that the CVZone model is not none. Test both models 
    def test_load_cvzone(self):
        self.assertIsNotNone(CVZONE_DETECTOR, "CVZone model could not be loaded")
        self.assertIsNotNone(CVZONE_DETECTOR_MAX_ONE, "CVZone model could not be loaded")
    


# Runs all unit tests within this file 
if __name__ == '__main__':
    unittest.main()