from constants import CVZONE_DETECTOR_MAX_ONE, EYE_DISTANCE_INDEX, FOCAL_LENGTH, INTEROCULAR_DISTANCE

def estimate_depth(landmarks: list[list[int]]):
    """Estimate the Z-coordinate (depth) for a detected face.

    This function calculates the depth, which is the distance between the screen and the user, using a method that relies on the distance between the eyes. 
    It uses the focal length and the average distance between the eyes, to estimate the depth based on eye landmarks detected. 

    Parameters:
        landmarks (list[list[int]]): A list of arrays, each array representing a landmark with x and y position of that landmark.

    Returns:
        int: The distance between the user and the camera 
    
    """
    # Check that the list has the 468 landmarks 
    if len(landmarks) != 468:
        print("ERROR: Invalid length of landmark list expected 468, was {len(landmarks)}")
        return None 
    
    # Retrieve the eye indexes 
    left_eye = landmarks[EYE_DISTANCE_INDEX['left_eye']]
    right_eye = landmarks[EYE_DISTANCE_INDEX['right_eye']]

    # Calculate distance between eyes
    w, _ = CVZONE_DETECTOR_MAX_ONE.findDistance(left_eye, right_eye)
    
    # Estimate depth
    return int((INTEROCULAR_DISTANCE * FOCAL_LENGTH) / w)
   