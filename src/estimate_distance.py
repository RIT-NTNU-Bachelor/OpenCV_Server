from constants.model_constants import CVZONE_DETECTOR_MAX_ONE, EYE_DISTANCE_INDEX
from constants.config_constants import FOCAL_LENGTH, INTEROCULAR_DISTANCE


def get_z_estimation(face):
    """
    Estimate the Z-coordinate (depth) for a detected face using the distance between the eyes.
    It uses the CVZone 

    Parameters:
    - face: A list representing a detected face with specific indices for eyes.

    Returns:
    - d: The estimated depth (Z-coordinate).
    """
    try:
        # Accessing eye positions
        if EYE_DISTANCE_INDEX['left_eye'] < len(face) and EYE_DISTANCE_INDEX['right_eye'] < len(face):
            leftEye = face[EYE_DISTANCE_INDEX['left_eye']]
            rightEye = face[EYE_DISTANCE_INDEX['right_eye']]
        else:
            raise IndexError("Eye indices are out of the range of the face array.")

        # Calculate pixel distance between eyes
        w, _ = CVZONE_DETECTOR_MAX_ONE.findDistance(leftEye, rightEye)
        
        # Estimate depth
        d = int((INTEROCULAR_DISTANCE * FOCAL_LENGTH) / w)
        return d
    except Exception as e:
        print(f"ERROR: An error occurred while trying to estimate Z: {e}")
        return None
