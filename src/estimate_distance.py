from constants import CVZONE_DETECTOR_MAX_ONE, EYE_DISTANCE_INDEX, FOCAL_LENGTH, INTEROCULAR_DISTANCE

def estimate_depth(landmarks: list):
    """ Estimate the Z-coordinate (depth) for a detected face. Depth is the distance between the screen and the user. 
    
    It uses the CVZones distance estimation. Using the distance between the eyes, focal length and a known average distance between the eyes.

    Args: 
    - landmarks(list): A list representing a detected face with all the 468 landmarks.

    Returns:
    - d(int | None): The estimated depth (Z-coordinate), or none if the incorrect list of landmarks was given. 
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
   