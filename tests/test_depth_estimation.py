# Set up project path for the test
from tests.test_utils import set_project_path_for_tests, get_images_from_dataset, save_test
set_project_path_for_tests()

# Import unittest package for running unit tests 
import unittest

# Get the estimated depth function 
from estimate_distance import estimate_depth  

# Import the function that has 
from models.code.cvzone import detect_face_cvzone

# Import the constant for the 
from constants.model_constants import CVZONE_DETECTOR_MAX_ONE


# Constants for testing with the IAS Lab RGB-D Face Dataset 
PATH_TO_DATASET = "./data/test_data/IAS-Lab_RGB-D_Face_Dataset_subset/"
PATH_TO_ONE_METER_DATA = PATH_TO_DATASET + "1m/"
PATH_TO_TWO_METER_DATA = PATH_TO_DATASET + "2m/"

FOCAL_LENGTH_IN_DATASET = 1056.244

class TestDepthEstimation(unittest.TestCase):
    
    def setUp(self):
        # Retrieve all the images for testing
        # Retrieving the images where the user is one and two meters away
        (one_meter_images, one_meter_images_width) = get_images_from_dataset(PATH_TO_ONE_METER_DATA)
        (two_meter_images, two_meter_images_width) = get_images_from_dataset(PATH_TO_TWO_METER_DATA)

        self.faces_one_meter = one_meter_images
        self.face_one_meter_width = one_meter_images_width
        self.faces_two_meters = two_meter_images

        # Checking that the setup was successful 
        self.assertEqual(len(self.faces_one_meter), 6)
        self.assertEqual(len(self.faces_two_meters), 6)

    def test_depth_one_meter(self):
        count = 0

        # For each image evaluate the distance for each image where the participant is one meter away
        for image in self.faces_one_meter:
            # Get the face that said cvzone 

            face = detect_face_cvzone(image,CVZONE_DETECTOR_MAX_ONE, detectMultipleFaces=False)
            if face == None:
                count += 1
                print("[INFO]: Skipping image")
                continue

                

            # Check that the face is found
            # Note that the testing for that a face detection works are found in test_models_one_face.py
            self.assertIsNotNone(face, "ERROR: could not find face in image")
            print(face)

            img_width = image[2]

            # Get the depth and test that it is 1 meter = 100 cm 
            depth = estimate_depth(face, self.face_one_meter_width, focal_length=FOCAL_LENGTH_IN_DATASET)
            self.assertIsNotNone(depth, "ERROR: depth was NONE for a image with the ground truth of 1 meter")
            self.assertEqual(depth, 100, "ERROR: depth did not correlate with the ground truth of 1 meter")

        print("Images skipped ", count)







            

        


# Run the tests
if __name__ == '__main__':
    unittest.main()